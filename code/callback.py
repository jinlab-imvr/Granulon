import logging
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)

class ProjectorGradMonitor(TrainerCallback):

    def _log_projector_grads(self, tag: str, model, step: int):
        total_norm_sq = 0.0
        n_params = 0
        for n, p in model.named_parameters():
            if "multi_modal_projector" in n:
                logger.info(f"[{tag}] {n}: requires_grad={p.requires_grad}, grad_is_none={p.grad is None}, shape={tuple(p.shape)}")
            if "multi_modal_projector" in n and p.grad is not None:
                g = p.grad.detach()
                total_norm_sq += float(g.norm(2).item() ** 2)
                n_params += 1
        if n_params:
            logger.info(f"[{tag}] step={step} projector grad norm={(total_norm_sq ** 0.5):.6f}")
        else:
            logger.warning(f"[{tag}] step={step} projector with no grad！")

    def on_substep_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        if state.global_step == 0 or state.global_step % 50 == 0:
            logger.info(f"=== on_substep_end: global_step={state.global_step}, total_flos={state.total_flos} ===")
            self._log_projector_grads("substep_end", model, state.global_step)

    def on_pre_optimizer_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        if state.global_step == 0 or state.global_step % 50 == 0:
            logger.info(f"=== on_pre_optimizer_step: global_step={state.global_step}, total_flos={state.total_flos} ===")
            self._log_projector_grads("pre_optimizer_step", model, state.global_step)

    def on_optimizer_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        if state.global_step == 0 or state.global_step % 50 == 0:
            logger.info(f"=== on_optimizer_step: global_step={state.global_step}, total_flos={state.total_flos} ===")
            self._log_projector_grads("optimizer_step", model, state.global_step)