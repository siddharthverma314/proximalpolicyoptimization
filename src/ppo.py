import torch

def ppo_clip_loss(policy, arc, epsilon):
    """
    Defines a PPO loss
    :param policy: a policy.Policy subclass
    :param arc: a data_collection.Arc subclass
    :returns: loss as a tensor
    """
    # simplification taken from OpenAI spinning up
    r = policy(arc.states).prob(arc.actions) / arc.probs
    policy_factor = r * arc.advantages

    # compute g
    g = ((arc.advantages >= 0).float() * 2  - 1) * epsilon + 1
    g = (g * arc.advantages).detach()

    return torch.min(policy_factor, g).mean()


def vpg_loss(policy, arc):
    """Defines a VPG loss"""
    probs = policy(arc.states).log_prob(arc.actions)
    return (probs * arc.advantages).mean()


def optimize_policy(policy, policy_optim,
                    arc, log,
                    policy_iter=100,
                    max_kl_divergence=10,
                    epsilon=0.1,
                    use_vpg=False):
    log.log("Optimizing policy...")
    previous_policy = policy(arc.states)
    first_loss = None

    for i in range(policy_iter):
        if not use_vpg:
            loss = ppo_clip_loss(policy, arc, epsilon)
        else:
            loss = vpg_loss(policy, arc)

        if not first_loss:
            first_loss = loss
        log.log(f"Policy Loss: {loss.item()}")
        policy_optim.zero_grad()
        (-loss).backward()
        policy_optim.step()
        kld = policy(arc.states).kl_divergence(previous_policy)
        log.log(f"KL Divergence: {kld}")

        if not first_loss:
            first_loss = loss
        if kld > max_kl_divergence:
            break

    data = {
        'policy_loss_first': first_loss.item(),
        'policy_loss_last': loss.item(),
        'policy_loss_iterations': i + 1,
        'policy_loss_kld': kld.item()
    }
    log.update(data)


def value_loss(value_fn, arc):
    v = value_fn(arc.states).squeeze()
    dot = (arc.rewards_to_go - v)**2
    loss = dot.mean()
    return loss


def optimize_value(value_fn, value_optim, arc, log, value_iter=100):
    log.log("Optimizing value...")
    first_loss = None
    for i in range(value_iter):
        loss = value_loss(value_fn, arc)
        if not first_loss:
            first_loss = loss
        log.log(f"Value Loss: {loss.item()}")
        value_optim.zero_grad()
        loss.backward()
        value_optim.step()

    data = {
        'value_loss_first': first_loss.item(),
        'value_loss_last': loss.item(),
        'value_loss_iterations': i + 1
    }
    log.update(data)

