import torch
import torch.optim as optim


def create_scheduler(optimizer, scheduler_type='step', scheduler_params=None, num_epochs=None, steps_per_epoch=None):
    params = scheduler_params or {}
    st = scheduler_type.lower() if scheduler_type else 'step'
    if st == 'step':
        step_size = params.get('step_size', 10)
        gamma = params.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if st == 'exponential':
        gamma = params.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if st == 'cosine':
        T_max = params.get('T_max', num_epochs or 30)
        eta_min = params.get('eta_min', 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    if st == 'cosine_warm':
        T_0 = params.get('T_0', 10)
        T_mult = params.get('T_mult', 2)
        eta_min = params.get('eta_min', 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
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
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.get('step_size', 10), gamma=params.get('gamma', 0.5))



def initialize_pointnet2_model(num_classes, device, normal_channel=True, learning_rate=0.001,
                               weight_decay=1e-4, step_size=10, gamma=0.5, scheduler_type='step',
                               scheduler_params=None, num_epochs=None, steps_per_epoch=None):
    from pointnet2_cls import get_model, get_loss

    model = get_model(num_class=num_classes, normal_channel=normal_channel).to(device)
    criterion = get_loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = create_scheduler(optimizer, scheduler_type=scheduler_type, scheduler_params=scheduler_params,
                                 num_epochs=num_epochs, steps_per_epoch=steps_per_epoch)

    return model, criterion, optimizer, scheduler
