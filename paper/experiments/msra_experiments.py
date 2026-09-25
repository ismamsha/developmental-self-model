"""
Reproducible experiment harness for the MSRA paper.

Runs the agents defined in ../../main.py (unmodified) plus ablations and
baselines, across seeds and difficulty levels, and writes per-episode and
per-step records to ../results/.

Notes
-----
* main.py imports a module `helpers` (EmergencyResponse, ResourceBuffer) that is
  not part of the repository. We inject a minimal stub: EmergencyResponse never
  forces an action and ResourceBuffer is an inert container. All emergency
  behaviour that remains is therefore the rule layer inside
  MetacognitiveSelfAgent.select_action().
* Console output produced by main.py is suppressed.

Usage:
    python msra_experiments.py --episodes 200 --seeds 0 1 2 3 4 \
        --difficulties normal hard expert --variants all --jobs 4
"""

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
import types
from multiprocessing import Pool

import numpy as np

# ---------------------------------------------------------------------------
# Import main.py with a stub for the missing `helpers` module
# ---------------------------------------------------------------------------
# Directory containing the original main.py (the MSRA repository root). Set MSRA_REPO
# when running from another checkout, e.g. the developmental-self-model repository.
REPO_ROOT = os.path.abspath(os.environ.get(
    "MSRA_REPO", os.path.join(os.path.dirname(__file__), "..", "..")))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

_helpers = types.ModuleType("helpers")


class _EmergencyResponse:
    def check_emergency(self, self_state):
        return None


class _ResourceBuffer:
    pass


_helpers.EmergencyResponse = _EmergencyResponse
_helpers.ResourceBuffer = _ResourceBuffer
sys.modules["helpers"] = _helpers
sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import torch  # noqa: E402

import main as M  # noqa: E402

ACTION_NAMES = ["REST", "REPAIR", "WORK_HIGH", "WORK_MED", "WORK_LOW"]


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def _log_prob(agent, obs, state, a):
    with torch.no_grad():
        probs, _ = agent.actor_critic(obs.unsqueeze(0), state.to_tensor().unsqueeze(0))
    return torch.log(probs[0, a])


class NoShieldMSRA(M.MetacognitiveSelfAgent):
    """MSRA without the hand-written rule layer in select_action():
    only doubt-adjusted exploration + imagination (with caution-mode masking)."""

    def select_action(self, observation, current_state, explore=True):
        eps = self.exploration_rate * (1.0 + current_state.doubt_t * 2)
        if explore and random.random() < min(0.8, eps):
            if self.in_caution_mode:
                a = random.choice([0, 1, 4])
            elif current_state.doubt_t > 0.08:
                a = random.choices(range(self.action_dim), weights=[0.35, 0.35] + [0.1] * 3)[0]
            else:
                a = random.randint(0, self.action_dim - 1)
        else:
            _, a, _ = self.imagine_futures(observation, current_state)
        return int(a), _log_prob(self, observation, current_state, a)


class LearnedOnlyMSRA(NoShieldMSRA):
    """No rule layer AND no hand-coded 'expert economics' filter: imagination
    scores actions only with the learned self-model and the utility function.
    Caution-mode masking (doubt-driven) is kept."""

    def imagine_futures(self, observation, current_state):
        utils = []
        for a in range(self.action_dim):
            s = self.predict_next_state(observation, current_state, a)
            u, _ = self.compute_utility(0, s)
            utils.append(u)
        if self.in_caution_mode:
            allow = [0] + ([1] if current_state.d_t > 0.75 and current_state.r_t > 0.7 else [])
            best = allow[int(np.argmax([utils[i] for i in allow]))]
        else:
            best = int(np.argmax(utils))
        return utils, best, []


class LearnedOnlyNoDoubt(LearnedOnlyMSRA):
    """LearnedOnlyMSRA with the metacognitive channel removed (doubt = 0)."""

    def update_metacognition(self, predicted_tensor, actual_tensor, agent_reward=None):
        err = torch.abs(predicted_tensor - actual_tensor).mean().item()
        self.self_state.doubt_t = 0.0
        self.in_caution_mode = False
        return 0.0, err


class NoDoubtMSRA(M.MetacognitiveSelfAgent):
    """Full MSRA (with rule layer) but doubt clamped to 0 -> caution never fires."""

    def update_metacognition(self, predicted_tensor, actual_tensor, agent_reward=None):
        err = torch.abs(predicted_tensor - actual_tensor).mean().item()
        self.self_state.doubt_t = 0.0
        self.in_caution_mode = False
        return 0.0, err


class ShieldOnly(M.MetacognitiveSelfAgent):
    """The hand-written rule layer of MSRA with no learning in the loop:
    when no rule fires, act with the agent's own dynamic_work_intensity()
    heuristic instead of imagination. No exploration, doubt fixed at 0."""

    def imagine_futures(self, observation, current_state):
        a = self.dynamic_work_intensity(current_state.r_t, current_state.d_t)
        return [0.0] * self.action_dim, a, []

    def select_action(self, observation, current_state, explore=True):
        return super().select_action(observation, current_state, explore=False)

    def update_metacognition(self, predicted_tensor, actual_tensor, agent_reward=None):
        err = torch.abs(predicted_tensor - actual_tensor).mean().item()
        self.self_state.doubt_t = 0.0
        self.in_caution_mode = False
        return 0.0, err


class ActorCriticAgent(M.ProactiveSelfReferentialAgent):
    """Model-free baseline: samples from the actor-critic network defined in
    main.py and trains it with main.py's own train_models().
    main.py's actor loss does not detach the advantage and takes log(softmax),
    which can drive the network to NaN; we then fall back to a uniform policy
    and count the event (reported in the paper)."""
    nan_events = 0

    def select_action(self, observation, current_state, explore=True):
        with torch.no_grad():
            probs, _ = self.actor_critic(observation.unsqueeze(0),
                                         current_state.to_tensor().unsqueeze(0))
        if not torch.isfinite(probs).all():
            self.nan_events += 1
            return random.randint(0, self.action_dim - 1), torch.tensor(0.0)
        a = int(torch.multinomial(probs[0], 1).item())
        return a, torch.log(probs[0, a])


class RandomAgent(M.ProactiveSelfReferentialAgent):
    def select_action(self, observation, current_state, explore=True):
        return random.randint(0, self.action_dim - 1), torch.tensor(0.0)


class AlwaysRest(M.ProactiveSelfReferentialAgent):
    def select_action(self, observation, current_state, explore=True):
        return 0, torch.tensor(0.0)


class SelfGradedMSRA(M.MetacognitiveSelfAgent):
    """Reference-free metacognition: the self-model is graded (and trained)
    against a target in which the externally measured variables (load,
    resources, degradation) are replaced by the self-model's own prediction.
    Control still sees the true state; only the evaluator loses its anchor."""
    reference_free = True


class SelfGradedLearnedOnly(LearnedOnlyMSRA):
    reference_free = True


VARIANTS = {
    "msra_full": M.MetacognitiveSelfAgent,
    "msra_no_doubt": NoDoubtMSRA,
    "msra_no_shield": NoShieldMSRA,
    "learned_only": LearnedOnlyMSRA,
    "learned_only_no_doubt": LearnedOnlyNoDoubt,
    "shield_only": ShieldOnly,
    "proactive_base": M.ProactiveSelfReferentialAgent,
    "actor_critic": ActorCriticAgent,
    "random": RandomAgent,
    "always_rest": AlwaysRest,
    "self_graded_full": SelfGradedMSRA,
    "self_graded_learned_only": SelfGradedLearnedOnly,
}


# ---------------------------------------------------------------------------
# Episode loop (mirrors main.train_metacognitive_agent)
# ---------------------------------------------------------------------------
def run(variant, difficulty, seed, episodes, record_steps_every=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    env = M.HomeostasisEnv(obs_dim=10, difficulty=difficulty)
    agent = VARIANTS[variant](obs_dim=10, action_dim=5)
    metacog = isinstance(agent, M.MetacognitiveSelfAgent)
    ref_free = getattr(agent, "reference_free", False)

    ep_rows, step_rows = [], []
    survival_history = []
    t0 = time.time()
    sink = io.StringIO()
    for ep in range(episodes):
        obs = env.reset()
        agent.self_state = M.SelfState(x_t=0.0, r_t=1.0, c_t=0.5, g_t=0.7, d_t=0.0, doubt_t=0.0)
        if metacog:
            agent.in_caution_mode = False
            agent.caution_mode_duration = 0
        ep_reward, steps, caution_steps, shocks = 0.0, 0, 0, 0
        actions = [0] * 5
        cause = "survived"
        errs = []
        while True:
            with contextlib.redirect_stdout(sink):
                a, _ = agent.select_action(obs, agent.self_state, explore=True)
            sink.seek(0)
            sink.truncate()
            a = int(a)
            next_obs, reward, load, done, status, shocked = env.step(a)
            actions[a] += 1
            shocks += int(shocked)
            if metacog:
                agent.shock_history.append(1 if shocked else 0)
                agent.consecutive_rests = agent.consecutive_rests + 1 if a == 0 else 0

            pred = agent.predict_next_state(obs, agent.self_state, a).to_tensor()
            actual = torch.tensor([load, env.r, agent.self_state.c_t, agent.self_state.g_t,
                                   env.d, agent.self_state.doubt_t], dtype=torch.float32)
            # Error against reality is always logged, whatever the agent uses.
            anchored_err = torch.abs(pred - actual).mean().item()
            target = actual
            if ref_free:
                target = actual.clone()
                target[[0, 1, 4]] = pred[[0, 1, 4]]

            if metacog:
                doubt, err = agent.update_metacognition(pred, target, agent_reward=reward)
                agent.update_self_state(load, env.r, env.d, err)
            else:
                err = torch.abs(pred - target).mean().item()
                agent.prediction_errors.append(err)
                agent.update_self_state(load, env.r, env.d)
            errs.append(anchored_err)

            agent.store_experience(obs, agent.self_state.to_tensor(), a, reward, next_obs, target,
                                   pred, target, done)
            if steps % 5 == 0 and len(agent.memory) >= 32:
                agent.train_models(batch_size=32)

            in_caution = bool(getattr(agent, "in_caution_mode", False))
            caution_steps += int(in_caution)
            if record_steps_every and ep % record_steps_every == 0:
                step_rows.append(dict(ep=ep, t=steps, a=a, r=env.r, d=env.d, reward=reward,
                                      shocked=int(shocked), doubt=agent.self_state.doubt_t,
                                      err_used=err, err_anchored=anchored_err,
                                      caution=int(in_caution),
                                      beta=agent.compute_effective_beta(agent.self_state)))
            ep_reward += reward
            steps += 1
            obs = next_obs
            if done or steps >= 100:
                if env.r <= 0:
                    cause = "resources"
                elif env.d >= 1.0:
                    cause = "degradation"
                break

        survived = int(cause == "survived")
        survival_history.append(survived)
        if metacog:
            agent.adapt_learning_rates(survival_history)
        ep_rows.append(dict(ep=ep, survived=survived, cause=cause, steps=steps,
                            reward=ep_reward, final_r=env.r, final_d=env.d,
                            caution_pct=100.0 * caution_steps / steps, shocks=shocks,
                            actions=actions, mean_err=float(np.mean(errs))))
    return dict(variant=variant, difficulty=difficulty, seed=seed, episodes=ep_rows,
                steps=step_rows, wall=time.time() - t0,
                nan_events=getattr(agent, "nan_events", 0))


def _job(args):
    variant, difficulty, seed, episodes, rec = args
    fn = os.path.join(RESULTS_DIR, "raw", f"{variant}__{difficulty}__s{seed}.json")
    if os.path.exists(fn):
        return f"{variant:26s} {difficulty:7s} seed={seed} (exists, skipped)"
    out = run(variant, difficulty, seed, episodes, rec)
    fn = os.path.join(RESULTS_DIR, "raw", f"{variant}__{difficulty}__s{seed}.json")
    with open(fn, "w") as f:
        json.dump(out, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    surv = np.mean([e["survived"] for e in out["episodes"]])
    return f"{variant:26s} {difficulty:7s} seed={seed} surv={surv:.3f} ({out['wall']:.0f}s)"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--difficulties", nargs="+", default=["normal", "hard", "expert"])
    p.add_argument("--variants", nargs="+", default=["all"])
    p.add_argument("--record-steps-every", type=int, default=10)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    variants = list(VARIANTS) if a.variants == ["all"] else a.variants
    os.makedirs(os.path.join(RESULTS_DIR, "raw"), exist_ok=True)
    jobs = [(v, d, s, a.episodes, a.record_steps_every)
            for d in a.difficulties for v in variants for s in a.seeds]
    with Pool(a.jobs) as pool:
        for line in pool.imap_unordered(_job, jobs):
            print(line, flush=True)
