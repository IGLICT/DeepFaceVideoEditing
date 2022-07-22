import jittor as jt

l2_criterion = jt.nn.MSELoss()

def l2_loss(real_images, generated_images):
    loss = l2_criterion(real_images, generated_images)
    return loss

