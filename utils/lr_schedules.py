import math

"""
These schedules are to be used in a torch.optim.lr_scheduler.LambdaLR
"""

# adapted from @karpathy
def cosine_warmup_schedule(lr, lr_min, warmup_iters, num_iters, start_iter=0):
    def schedule(iter):
        iter = start_iter + iter

        # linear warmup
        if iter < warmup_iters:
            return iter/warmup_iters
        
        # constant lr after num_iters
        if iter > num_iters:
            return lr_min/lr
        
        # in between, cosine decay
        decay_ratio = (iter - warmup_iters) / (num_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return (lr_min + coeff * (lr - lr_min))/lr
    
    return schedule

# WSD paper, eq 1
def wsd_schedule(warmup_iters, decay_iters, num_iters, start_iter=0):
    def schedule(iter):
        iter = start_iter + iter
        
        if iter > num_iters:
            return 0.
        
        # linear warmup
        if iter < warmup_iters:
            return iter/warmup_iters
        
        # hold
        elif iter < (num_iters - decay_iters):
            return 1.
        
        # decay (with 1-sqrt)
        else:
            if decay_iters==0:
                return 1.
            return 1 - math.sqrt((iter - (num_iters - decay_iters))/decay_iters)
    return schedule
