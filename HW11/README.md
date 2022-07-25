# W251 HW11 DA QI REN

## What

In this homework, I trained a Lunar Lander module to land properly using Jetson Nano 4G. Use Reinforcement Learning and tweak the model parameters to improve the model results.

#### My modifications 

(1) Configuration for PQN parameters 

        #######################
        # Change these parameters to improve performance
        self.density_first_layer = 64
        self.density_second_layer = 32
        self.density_thired_layer = 16
        self.num_epochs = 1
        self.batch_size = 256
        self.epsilon_min = 0.01

        # epsilon will randomly choose the next action as either
        # a random action, or the highest scoring predicted action
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.99

        # Learning rate
        self.lr = 0.001

        #######################

(2) Adding a new layer to the original model: 

        model = Sequential()
        model.add(Dense(self.density_first_layer, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(self.density_second_layer, activation=relu))
        model.add(Dense(self.density_thired_layer, activation=relu))
        model.add(Dense(self.num_action_space, activation=linear))


##### Training Process: 

1995    : Episode || Reward:  303.1633172527662         || Average Reward:  253.1940158838653    epsilon:  0.00998645168764533
1996    : Episode || Reward:  195.16712949470258        || Average Reward:  252.71620855445806   epsilon:  0.00998645168764533
1997    : Episode || Reward:  320.4031225054513         || Average Reward:  255.55198799757997   epsilon:  0.00998645168764533
1998    : Episode || Reward:  285.6891690700164         || Average Reward:  256.02868412792424   epsilon:  0.00998645168764533
1999    : Episode || Reward:  268.1372233505265         || Average Reward:  256.01345174663055   epsilon:  0.00998645168764533


![episode0](episode0.gif)
![episode0](episode1000.gif)
![episode0](episode1900.gif)


#### Testing Process

96      : Episode || Reward:  291.718547261176
97      : Episode || Reward:  243.1428731416763
98      : Episode || Reward:  249.66159692830578
99      : Episode || Reward:  290.74481702270094
Average Reward:  260.1056624914775
Total tests above 200:  95

real    4m37.796s
user    0m0.140s
sys     0m0.088s

![testing_run20.gif](testing_run20.gif)
![testing_run40.gif](testing_run40.gif)
![testing_run60.gif](testing_run60.gif)

 
