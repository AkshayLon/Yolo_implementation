import torch, itertools, math
from torch import nn

class yolov1(nn.Module):

    def __init__(self):
        super(yolov1, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.third_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        layers_4 = list()
        layers_4.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        layers_4.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        for i in range(3):
            layers_4.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1))
            layers_4.append(nn.LeakyReLU(negative_slope=0.1))
            layers_4.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
            layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        layers_4.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        layers_4.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        self.fourth_layer = nn.Sequential(*layers_4)
        self.fifth_layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.sixth_layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=50176, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=539),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(padding=1, stride=2, kernel_size=2)

    def forward(self, x):
        o1 = self.pool(self.first_layer(x))
        o2 = self.pool(self.second_layer(o1))
        o3 = self.pool(self.third_layer(o2))
        o4 = self.pool(self.fourth_layer(o3))
        o5 = self.fifth_layer(o4)
        o6 = self.sixth_layer(o5)
        x = self.fc(o6)

        return x

class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, outputs, target):
        """
        outputs = Tensor of size [batch_size,539]
        target = Tensor of size [batch_size,2,4] with each entry [x,y,w,h]
        """
        def obj_presence(grid_number, box_centre):
            x_lower, y_lower = (grid_number%7)*(1/7), int(grid_number/7)*(1/7)
            x_upper, y_upper = x_lower+(1/7), y_lower+(1/7)
            if (x_lower <= box_centre[0] <= x_upper) and (y_lower <= box_centre[1] <= y_upper):
                return True
            return False


        def true_confidence(pred_info, true_info):
            x_p, y_p, w_p, h_p = [*pred_info]
            x_t, y_t, w_t, h_t = [*true_info]
            p_bounds, p_area = (x_p-(0.5*w_p), x_p+(0.5*w_p),y_p-(0.5*h_p), y_p+(0.5*h_p)), w_p*h_p
            t_bounds, t_area = (x_t-(0.5*w_t), x_t+(0.5*w_t),y_t-(0.5*h_t), y_t+(0.5*h_t)), w_t*h_t
            intersection_bounds = (
                max(p_bounds[0], t_bounds[0]),
                min(p_bounds[1], t_bounds[1]),
                max(p_bounds[2], t_bounds[2]),
                min(p_bounds[3], t_bounds[3])
            )
            intersection = abs(intersection_bounds[0]-intersection_bounds[1])*abs(intersection_bounds[2]-intersection_bounds[3])
            union = p_area + t_area - intersection
            return intersection/union

        lambda_c, lambda_n, cum_loss = 5, 0.5, 0
        batch_size = target.shape[0]
        for x in range(batch_size):
            current_output, current_target = outputs[x].reshape(49,11), target[x]
            num_boxes = 1 if (current_target[1][0]==-1) else 2
            one_i = set()
            for i,j in itertools.product(range(49),range(num_boxes)):
                one_ij = obj_presence(grid_number=i, box_centre=(current_target[j][0], current_target[j][1]))
                if one_ij:
                    one_i.add(i)
                    cum_loss += lambda_c*(((current_target[j][0]-current_output[i][5*j])**2)+((current_target[j][1]-current_output[i][(5*j)+1])**2))
                    cum_loss += lambda_c*(((math.sqrt(current_target[j][2])-math.sqrt(current_output[i][(5*j)+2]))**2)+((math.sqrt(current_target[j][3])-math.sqrt(current_output[i][(5*j)+3]))**2))
                    cum_loss += (true_confidence(pred_info=current_output[i][(5*j):(5*j)+4], true_info=current_target[j])-current_output[i][(5*j)+4])**2
                else:
                    cum_loss += lambda_n*(current_output[i][(5*j)+4]**2)
            for i in one_i:
                cum_loss += (current_output[i][-1]-sum(list(true_confidence(pred_info=current_output[i][(5*j):(5*j)+4], true_info=current_target[j]) for j in range(num_boxes))))**2
        return cum_loss
        

