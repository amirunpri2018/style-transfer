import argparse

from PIL import Image

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Style Transfer')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')
    parser.add_argument('--content-image', default='image/dancing.jpg', help='content image')
    parser.add_argument('--style-image', default='image/picasso.jpg', help='style image')
    parser.add_argument('--image-size', type=int, default=400, help='size of resized image')

    parser.add_argument('--style-weight', type=int, default=100, help='weight for style loss')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--num-steps', type=int, default=2000, help='size of resized image')
    parser.add_argument('--print-interval', type=int, default=50, help='frequency of print statements')
    parser.add_argument('--sample-interval', type=int, default=500, help='frequency of generating sample images')
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load content and style images
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[None])
    ])
    content_image = transform(Image.open(args.content_image)).to(device)
    style_image = transform(Image.open(args.style_image)).to(device)
    target_image = content_image.clone().requires_grad_(True)

    # Extract features from VGG19
    model = models.vgg19(pretrained=True).features.to(device).eval()
    def extract_features(input, layers=['0', '5', '10', '19', '28']):
        features = []
        for i, (name, module) in enumerate(model._modules.items()):
            input = module(input)
            if name in layers:
                features.append(input)
        return features

    optimizer = torch.optim.Adam([target_image], lr=args.lr, betas=[0.5, 0.999])

    for step in range(args.num_steps):
        content_features = extract_features(content_image)
        style_features = extract_features(style_image)
        target_features = extract_features(target_image)
        content_loss, style_loss = 0, 0

        for (target, content, style) in zip(target_features, content_features, style_features):
            # Compute content loss
            content_loss += torch.mean((target - content) ** 2)

            # Compute style loss via Gram matrices
            _, C, H, W = target.size()
            target = target.view(C, -1)
            target = target.matmul(target.t())
            style = style.view(C, -1)
            style = style.matmul(style.t())
            style_loss += torch.mean((target - style) ** 2) / (C * H * W)

        loss = content_loss + args.style_weight * style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % args.print_interval == 0:
            print('Step {} / {}: content_loss {:.4f} | style_loss {:.4f}'.format(
                step + 1, args.num_steps, content_loss.item(), style_loss.item()))

        if (step + 1) % args.sample_interval == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            image = denorm(target_image.clone().squeeze()).clamp_(0, 1)
            torchvision.utils.save_image(image, 'image/sample-{}.png'.format(step + 1))


if __name__ == '__main__':
    args = get_args()
    main(args)
