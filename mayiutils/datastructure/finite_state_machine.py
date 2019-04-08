#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:@hotmail.com
@file: finite_state_machine.py
@time: 2019/4/8 9:28
有限状态机（Finite-state machine, FSM），又称有限状态自动机，简称状态机，是表示有限个状态以及在这些状态之间的转移和动作等行为的数学模型。

transitions是一个由Python实现的轻量级的、面向对象的有限状态机框架。
https://github.com/pytransitions/transitions

transitions最基本的用法如下：

1.先自定义一个类Matter
2.定义一系列状态和状态转移（定义状态和状态转移有多种方式，官网上给了最快速理解的一个例子）
3.初始化状态机
4.获取当前的状态或者进行转化

"""
from transitions import Machine
# from transitions.extensions import GraphMachine as Machine
import random


class NarcolepticSuperhero(object):

    # Define some states. Most of the time, narcoleptic superheroes are just like
    # everyone else. Except for...
    states = ['asleep', 'hanging out', 'hungry', 'sweaty', 'saving the world']

    def __init__(self, name):

        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.
        self.name = name

        # What have we accomplished today?
        self.kittens_rescued = 0

        # Initialize the state machine
        self.machine = Machine(model=self, states=NarcolepticSuperhero.states, initial='asleep')

        # Add some transitions. We could also define these using a static list of
        # dictionaries, as we did with states above, and then pass the list to
        # the Machine initializer as the transitions= argument.

        # At some point, every superhero must rise and shine. hang out 闲逛
        self.machine.add_transition(trigger='wake_up', source='asleep', dest='hanging out')

        # Superheroes need to keep in shape. work out 锻炼
        self.machine.add_transition('work_out', 'hanging out', 'hungry')

        # Those calories won't replenish themselves!
        self.machine.add_transition('eat', 'hungry', 'hanging out')

        # Superheroes are always on call. ALWAYS. But they're not always
        # dressed in work-appropriate clothing. distress call 求救信号
        self.machine.add_transition('distress_call', '*', 'saving the world',
                         before='change_into_super_secret_costume')

        # When they get off work, they're all sweaty and disgusting. But before
        # they do anything else, they have to meticulously log their latest
        # escapades. Because the legal department says so.
        self.machine.add_transition('complete_mission', 'saving the world', 'sweaty',
                         after='update_journal')

        # Sweat is a disorder that can be remedied with water.
        # Unless you've had a particularly long day, in which case... bed time!
        self.machine.add_transition('clean_up', 'sweaty', 'asleep', conditions=['is_exhausted'])
        self.machine.add_transition('clean_up', 'sweaty', 'hanging out')

        # Our NarcolepticSuperhero can fall asleep at pretty much any time.
        self.machine.add_transition('nap', '*', 'asleep')

    def update_journal(self):
        """ Dear Diary, today I saved Mr. Whiskers. Again. """
        self.kittens_rescued += 1

    def is_exhausted(self):
        """ Basically a coin toss. """
        return random.random() < 0.5

    def change_into_super_secret_costume(self):
        print("Beauty, eh?")


class Matter(object):
    pass


class AModel(object):
    def __init__(self):
        self.sv = 0  # state variable of the model
        self.conditions = {  # each state
            'sA': 0,
            'sB': 3,
            'sC': 6,
            'sD': 0,
        }

    def poll(self):
        if self.sv >= self.conditions[self.state]:
            self.next_state()  # go to next state
        else:
            #getattr(object, name[, default]) -> value
            getattr(self, 'to_%s' % self.state)()  # enter current state again

    def on_enter(self):
        print('entered state %s' % self.state)

    def on_exit(self):
        print('exited state %s' % self.state)


if __name__ == '__main__':
    mode = 3
    if mode == 3:
        """
        https://www.jianshu.com/p/decf86e0e420
        """
        model = AModel()

        # init transitions model
        list_of_states = ['sA', 'sB', 'sC', 'sD']
        machine = Machine(model=model, states=list_of_states, initial='sA',
                          ordered_transitions=True, before_state_change='on_exit',
                          after_state_change='on_enter')

        # begin main
        for i in range(0, 10):
            print('iter is: ' + str(i) + " -model state is:" + model.state)
            model.sv = i
            model.poll()
    if mode == 2:
        """
        官网示例https://github.com/pytransitions/transitions
        """
        batman = NarcolepticSuperhero("Batman")
        print(batman.state)#asleep
        batman.wake_up()
        print(batman.state)#hanging out
        batman.nap()
        print(batman.state)#asleep
        # batman.clean_up()
        # print(batman.state)#transitions.core.MachineError: "Can't trigger event clean_up from state asleep!"
        batman.wake_up()
        print(batman.state)#hanging out
        batman.work_out()
        print(batman.state)#hungry
        print(batman.kittens_rescued)#0

        batman.distress_call()#Beauty, eh?
        print(batman.state)#saving the world
        batman.complete_mission()
        print(batman.state)#sweaty
        batman.clean_up()
        print(batman.state)#hanging out
        print(batman.kittens_rescued)#1

    if mode == 1:
        """
        fsm基本使用示例https://www.jianshu.com/p/decf86e0e420
        """
        model = Matter()

        # The states argument defines the name of states
        states = ['solid', 'liquid', 'gas', 'plasma']

        # The trigger argument defines the name of the new triggering method
        transitions = [
            {'trigger': 'melt', 'source': 'solid', 'dest': 'liquid'},
            {'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas'},
            {'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas'},
            {'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma'}]

        machine = Machine(model=model, states=states, transitions=transitions, initial='solid')

        # Test
        print(model.state)  # solid
        model.melt()
        print(model.state)  # liquid
        model.evaporate()
        print(model.state)  # gas
        #model.get_graph().draw('my_state_diagram.png', prog='dot')