import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class WindowsOptim(nn.Module):
    def __init__(self, window_size=5, use_SSIM=False):
        super(WindowsOptim, self).__init__()
        self.window_size = window_size
        self.height = 0
        self.width = 0
        self.window_index = np.transpose(np.array(np.meshgrid(np.arange(-window_size, window_size + 1, 2), np.arange(-window_size, window_size + 1, 2)))).reshape(-1, 2)
        self.use_SSIM = use_SSIM
        self.avg_pool2d = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = self.avg_pool2d(x)
        mu_y = self.avg_pool2d(y)
        sigma_x = self.avg_pool2d(x ** 2) - mu_x ** 2
        sigma_y = self.avg_pool2d(y ** 2) - mu_y ** 2
        sigma_xy = self.avg_pool2d(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def forward(self, move_input, target):
        original_target = target
        target = F.pad(target, (self.window_size, self.window_size, self.window_size, self.window_size), mode='constant')
        move_cost = torch.zeros_like(original_target)
        weight = torch.zeros_like(original_target)

        self.height = original_target.shape[2]
        self.width = original_target.shape[3]

        for item in self.window_index:
            target_item = target[:, :, self.window_size - item[0]: self.height + self.window_size - item[0], self.window_size - item[1]: self.width + self.window_size - item[1]]
            weight_item = torch.exp(-torch.abs(target_item - original_target) / 2)
            if not self.use_SSIM:
                move_cost += torch.abs(target_item - move_input) * weight_item
            else:
                move_cost += (0.85 * torch.abs(target_item - move_input) + 0.15 * self.SSIM(target_item, move_input)) * weight_item
            weight += weight_item
            torch.cuda.empty_cache()

        return move_cost / weight


class MonodepthLoss(nn.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=20, lr_w=1):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n
        self.window_optim = WindowsOptim()
        if self.n != 4:
            print("MonodepthLoss's n is not 4.")

        self.fovw = 82.5 * np.pi / 180
        self.fovh = 29.7 * np.pi / 180

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        # Disparity is passed in NCHW format with 1 channel
        x_shifts = disp[:, 0, :, :]  
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def apply_disparity_equi_v2(self, img, disp):
        batch_size, _, height, width = img.size()

        x_base = torch.linspace(-self.fovw / 2, self.fovw / 2, width).repeat(batch_size, height, 1).type_as(img)
        y_base = torch.linspace(-self.fovh / 2, self.fovh / 2, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

        x_base_onedim = x_base.view(-1)
        y_base_onedim = y_base.view(-1)
        x_shifts_onedim = disp[:, 0, :, :].view(-1)

        ny = torch.sin(y_base_onedim)
        nx = torch.cos(y_base_onedim) * torch.sin(x_base_onedim)
        nz = torch.cos(y_base_onedim) * torch.cos(x_base_onedim)

        nx_add = nx + x_shifts_onedim

        lat = torch.atan(nx_add / nz)
        lon = torch.asin(ny)

        x_wrap = (lat + self.fovw / 2) / self.fovw
        y_wrap = (lon + self.fovh / 2) / self.fovh
        flow_field = torch.stack([x_wrap, y_wrap], dim=-1).view(batch_size, height, width, 2)

        output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity_equi_v2(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity_equi_v2(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, gt):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = self.gradient_x(gt)
        image_gradients_y = self.gradient_y(gt)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        #weights_x = torch.ones_like(weights_x)
        #weights_y = torch.ones_like(weights_y)

        smoothness_x = [disp_gradients_x[i] * weights_x
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def forward(self, input_left, input_right, left, right):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """

        # Prepare disparities
        disp_left_est = [F.interpolate(torch.mean(item, dim=1, keepdim=True), size=(left.shape[2], left.shape[3])) for item in input_left]
        disp_right_est = [F.interpolate(torch.mean(item, dim=1, keepdim=True), size=(left.shape[2], left.shape[3])) for item in input_left]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est

        # Generate images
        right_est = [self.generate_image_right(left,
                    disp_left_est[i]) for i in range(self.n)]
        left_est = [self.generate_image_left(right,
                     disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_right_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_left_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right)
        #"""
        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left)) for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i] - right)) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i], left)) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i], right)) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n - 1)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n - 1)]
        image_loss = sum(image_loss_left + image_loss_right)
        #"""
        image_loss_left += [torch.mean(self.window_optim(left_est[i], left)) for i in range(self.n - 1, self.n)]
        image_loss_right += [torch.mean(self.window_optim(right_est[i], right)) for i in range(self.n - 1, self.n)]
        image_loss = sum(image_loss_left + image_loss_right)
        #"""
        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** (self.n - 1 - i)
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** (self.n - 1 - i)
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss
        image_loss = image_loss
        disp_gradient_loss = disp_gradient_loss
        lr_loss = lr_loss

        return loss, image_loss, disp_gradient_loss, lr_loss, right_est
