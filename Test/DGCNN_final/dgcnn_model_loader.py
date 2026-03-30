import torch
import torch.optim as optim
from types import SimpleNamespace


def create_scheduler(optimizer, scheduler_type='step', scheduler_params=None, num_epochs=None, steps_per_epoch=None):
    params = scheduler_params or {}
    st = (scheduler_type or 'step').lower()
    if st == 'step':
        step_size = params.get('step_size', 20)
        gamma = params.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if st == 'exponential':
        gamma = params.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if st == 'cosine':
        T_max = params.get('T_max', num_epochs or 30)
        eta_min = params.get('eta_min', 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    if st == 'reduce_on_plateau':
        mode = params.get('mode', 'min')
        factor = params.get('factor', 0.5)
        patience = params.get('patience', 5)
        min_lr = params.get('min_lr', 1e-6)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)
    if st == 'onecycle':
        if num_epochs is None or steps_per_epoch is None:
            return None
        total_steps = int(num_epochs * steps_per_epoch)
        max_lr = params.get('max_lr', 1e-2)
        pct_start = params.get('pct_start', 0.3)
        anneal_strategy = params.get('anneal_strategy', 'cos')
        div_factor = params.get('div_factor', 25.0)
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps,
                                                   pct_start=pct_start, anneal_strategy=anneal_strategy,
                                                   div_factor=div_factor)
    # default
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.get('step_size', 20), gamma=params.get('gamma', 0.5))


def initialize_model(num_classes: int, device, k: int = 20, emb_dims: int = 1024, dropout: float = 0.5,
                     learning_rate: float = 0.001, weight_decay: float = 1e-4,
                     scheduler_type: str = 'step', scheduler_params: dict = None,
                     num_epochs: int = None, steps_per_epoch: int = None):
    """Create DGCNN model, loss, optimizer and scheduler.

    Returns: (model, criterion, optimizer, scheduler, actual_kwargs)
    """
    from model import DGCNN_cls, get_loss

    args = SimpleNamespace(k=int(k), emb_dims=int(emb_dims), dropout=float(dropout))
    model = DGCNN_cls(args, output_channels=num_classes).to(device)
    criterion = get_loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = create_scheduler(optimizer, scheduler_type=scheduler_type, scheduler_params=scheduler_params,
                                 num_epochs=num_epochs, steps_per_epoch=steps_per_epoch)

    actual_kwargs = {'k': args.k, 'emb_dims': args.emb_dims, 'dropout': args.dropout}
    return model, criterion, optimizer, scheduler, actual_kwargs
