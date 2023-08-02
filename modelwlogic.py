import numpy as np
from math import log
import torch
from training_model import CustomNetwork
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


WHEEL_RATE = 0.3
WARNING_RATE = 1.2
THRESHOLD = 0.25
THRESHOLD2 = 0.1

wheel = l_torque = r_torque = 0
wheel_queue = [0, 0, 0, 0, 0]

channel = EngineConfigurationChannel()

env = UnityEnvironment(file_name='Road1/Prototype 1', side_channels=[channel])
channel.set_configuration_parameters(time_scale=3)

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0, :]

device = torch.device("cpu")

# model that is appropriate for trained model
model1 = CustomNetwork().to(device)
model2 = CustomNetwork().to(device)
model3 = CustomNetwork().to(device)

model1.load_state_dict(torch.load("base1.model"))  # base model 1
model1.eval()
model2.load_state_dict(torch.load("base2.model"))  # base model 2 + model for fence at end of road 3
model2.eval()
model3.load_state_dict(torch.load("s_obst.model"))  # model for small obstacles
model3.eval()

control_list = [0, 0, 0]

cmv=[]
cmv2=[]
cmv3=[]

def transform(sensors):
    value = list()
    value.append(sensors[9] / 20)
    value.append(sensors[8] / 20)
    value.append(sensors[10] / 20)
    value.append(sensors[6] / 20)
    value.append(sensors[7] / 20)

    return value


# list에 있는 값들의 평균을 weight에 적혀있는 가중치를 이용해 계산
def mean_of_list(li, weights=[0.33, 0.33, 0.33]):
    summation = 0
    for i in range(len(li)):
        summation = summation + li[i] * weights[i]

    return summation


def add_wheel(data, num=1):
    for i in range(num):
        wheel_queue.pop(0)
        wheel_queue.append(data)


def find_(sensors):
    li = list()
    small = 987654321
    idx = 987654321
    for i in range(len(sensors)):
        if small > sensors[i]:
            small = sensors[i]
            idx = i
        if sensors[i] < THRESHOLD:
            li.append(1)
        else:
            li.append(0)

    return li, small, idx


def find_2(sensors):
    if sensors[0] < sensors[1]:
        if sensors[3] < sensors[4]:
            return sensors[0], sensors[1], sensors[3], sensors[4]
        else:
            return sensors[0], sensors[1], sensors[4], sensors[3]
    else:
        if sensors[3] < sensors[4]:
            return sensors[1], sensors[0], sensors[3], sensors[4]
        else:
            return sensors[1], sensors[0], sensors[4], sensors[3]



for i in range(100000):

    model_type="model 1"
    s=[]
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0, :]  #


    s.append(cur_obs[9] * cur_obs[8] / 400)
    a = min(cur_obs[8], cur_obs[10], cur_obs[6])
    s.append(a/20)
    s.append(cur_obs[6] * cur_obs[7] / 400)

    ###########chose appropriate model to use ################

    input = torch.from_numpy(cur_obs[6:11])
    output = model1(input)

    if cur_obs[6] > 19 and cur_obs[7] > 15.8 and cur_obs[8] > 19 and cur_obs[9] > 9 and cur_obs[9] < 13 and cur_obs[10] > 18:
        output = model2(input)
        model_type="model 2"
        cmv2.append(0)
    else:
        cmv2.append(1)
        if i > 70:
            for j in range(1, 70):
                if cmv2[i - j] == 0:
                    output = model2(input)
                    model_type = "model 2"

    if s[1] + s[0] < 0.75 and s[2] < 0.65 and s[2] > 0.35:
        output = model3(input)
        model_type = "model 3"
        cmv3.append(0)

    else:
        cmv3.append(1)
        if i>15:
            for j in range(1,15):
                if cmv3[i-j]==0:
                    output = model3(input)
                    model_type = "model 3"

    driving_type = model_type
    arr = output.detach().numpy()
    ###################################################

    ################ logic part ################

    value = transform(cur_obs)

    l_mean = mean_of_list(value[:2], [0.7, 0.3])
    r_mean = mean_of_list(value[3:], [0.3, 0.7])

    # 왼쪽, 오른쪽 센서중 아주 가까운(0에 근접한) 것이 있는지 확인하기 위해 왼쪽의 두개 센서, 오른쪽의 두개 센서를 곱해서 확인함
    l_mean2 = value[0] * value[1]
    r_mean2 = value[3] * value[4]
    m_mean2 = value[1] * value[2] * value[3]

    flag, extreme, idx = find_(value)

    # 물체가 바로 앞에 있을 때 (자동차의 정면에 커다란 장애물이 있는 경우)
    if value[1] * value[2] * value[3] < THRESHOLD2:
        if value[0] < value[4]:
            add_wheel(0.99, 4)
        else:
            add_wheel(-0.99, 4)

    # 물체가 센서의 한두개 정도에만 감지 될 경우 (자동차의 측면, 정면에 작은 장애물이 있는 경우)
    elif flag != [0, 0, 0, 0, 0]:
        if value[1] < THRESHOLD and value[2] < THRESHOLD and value[3] < THRESHOLD:
            if value[0] < value[4]:
                add_wheel(0.99, 4)
            else:
                add_wheel(-0.99, 4)

        elif idx < 2 and l_mean2 < r_mean2:
            add_wheel(0.99, 4)
        elif idx > 2 and r_mean2 < l_mean2:
            add_wheel(-0.99, 4)
        else:
            if l_mean2 < r_mean2:
                add_wheel(0.99, 4)
            else:
                add_wheel(-0.99, 4)
        l_torque = 100 + 50 * r_mean
        r_torque = 100 + 50 * l_mean

    # 위의 경우가 아닌 경우 (장애물이 없는 코스들)
    else:
        wheel_queue.pop(0)
        wheel_queue.append(log(r_mean2 / l_mean2, 4) * WHEEL_RATE)
        l_torque = 100 + 50 * r_mean
        r_torque = 100 + 50 * l_mean

    #print(wheel_queue)

    # �
    wheel = mean_of_list(wheel_queue, [0.2, 0.2, 0.2, 0.2, 0.2])
    #print("car obs:  ", value)
    #print("car behavior:  ", wheel, l_torque, r_torque)

    ############################################################

    ###### determining driving type to use (model/logic) ######

    dis1 = 7.1
    dis2 = 6.2

    close_sensors = 0

    if cur_obs[6] < dis1:
        close_sensors+=1
    if cur_obs[7] < dis2:
        close_sensors+=1
    if cur_obs[8] < dis1:
        close_sensors+=1
    if cur_obs[9] < dis2:
        close_sensors+=1
    if cur_obs[10] < dis1:
        close_sensors+=1

    if wheel > arr + 0.3:
        wheel = arr + 0.3
    if wheel < arr - 0.3:
        wheel = arr - 0.3


    control_list = [arr, 137, 137]

    if close_sensors > 0:
        cmv.append(0)
        control_list = [wheel, 137, 137]
        driving_type="logic"
    else:
        cmv.append(1)
        if i>3:
            for j in range(1,3):
                if cmv[i-j]==0:
                    control_list = [wheel, 150, 150]
                    driving_type = "logic"
    ########################################################

    ######### goes straight forward when goal is near #########
    distance = (pow(cur_obs[0] - cur_obs[3], 2) + pow(cur_obs[2] - cur_obs[5], 2)) ** 0.5
    #print(distance)
    if distance < 10:
        control_list = [0, 137, 137]
    ############################################################

    print("driving with " + driving_type)
    # print(control_list)

    # Set the actions
    env.set_actions(behavior_name, np.array([control_list]))
    # Move the simulation forward
    env.step()

env.close()