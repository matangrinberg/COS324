import numpy as np
import numpy.random as npr
import sys

from SwingyMonkey import SwingyMonkey

c = 0
class Learner:

  def __init__(self):
    self.last_state  = { 'score': 0, 'tree': {'dist': 0, 'top': 0, 'bot': 0}, 'monkey': {'vel': 0, 'top': 0, 'bot': 0}}
    self.last_action = 0
    self.last_reward = 0
    self.q           = np.zeros([16, 16, 5, 2])
    self.rate        = np.ones([16, 16, 5, 2])
    self.gamma       = 0.1
    self.totscore    = 0
    self.hscore      = 0
    self.eps         = 1

  def eps_incr(self):
    self.eps = 1/np.power((np.power(1/self.eps, 1/2) +1), 2)

  def total_score(self):
    return self.totscore

  def getscore(self):
    return self.last_state.get('score')

  def high_score(self):
    return self.hscore

  def reset(self):
    self.last_state = {'score': 0, 'tree': {'dist': 0, 'top': 0, 'bot': 0}, 'monkey': {'vel': 0, 'top': 0, 'bot': 0}}
    self.last_action = 0
    self.last_reward = 0

  def action_callback(self, state):
    """Implement this function to learn things and take actions.
    Return 0 if you don't want to jump and 1 if you do."""
    # You might do some learning here based on the current state and the last state.

    # we ignore monkey top because it does not provide unique info (monkey bot suffices)

    scale_factor = 50
    vel_factor = 20
    dist_factor = 1.5 * scale_factor

    # Old State
    old_tree       = self.last_state.get('tree')
    old_monkey = self.last_state.get('monkey')

    old_tree_dist  = int(np.floor(old_tree.get('dist')/(dist_factor/2)))
    old_heightdiff   = int(np.floor((old_tree.get('bot') - old_monkey.get('bot'))/scale_factor)) + 8
    old_monkey_vel = int(np.floor(old_monkey.get('vel')/vel_factor))+3


    # New State
    tree           = state.get('tree')
    monkey         = state.get('monkey')

    tree_dist      = int(np.floor(tree.get('dist') / (dist_factor/2)))
    heightdiff     = int(np.floor((tree.get('bot') - monkey.get('bot'))/scale_factor)) + 8
    monkey_vel     = int(np.floor(monkey.get('vel') /vel_factor))+3

    max_val = np.max([self.q[tree_dist, heightdiff, monkey_vel, :]])


    opt_action = np.argmax([self.q[tree_dist, heightdiff, monkey_vel, :]])

    a = self.last_action

    self.q[old_tree_dist, old_heightdiff, old_monkey_vel, a] += \
        self.rate[old_tree_dist, old_heightdiff, old_monkey_vel, a]*((self.last_reward + self.gamma * max_val)  \
                   - self.q[old_tree_dist, old_heightdiff, old_monkey_vel, a])


    self.rate[old_tree_dist, old_heightdiff, old_monkey_vel, a] = \
      1/(1/self.rate[old_tree_dist, old_heightdiff, old_monkey_vel, a] + 1)


    new_action = opt_action
    eps_action = (np.random.randint(2) == True)
    test = np.random.rand()
    if test < self.eps:
      new_action = eps_action
      # print(test)
      print('random!')


    new_state = state

    if state.get('score') > self.hscore:
      self.hscore = state.get('score')
    self.totscore   += state.get('score')-self.last_state.get('score')
    self.last_action = new_action
    self.last_state  = new_state
    return self.last_action


  def reward_callback(self, reward):
    """This gets called so you can see what reward you get."""

    self.last_reward = reward

iterations = 130
learner = Learner()
fifties = 0
scores = np.zeros(iterations)

for i in range(iterations):
  if i % 20 == 0:
    fifties = learner.total_score()

  average = learner.total_score()/(i+1)
  fiftyaverage = (learner.total_score() - fifties)/(i%20 + 1)

  # Make a new monkey object.
  swing = SwingyMonkey(sound=False,            # Don't play sounds.
                       text="Epoch %d" % i,    # Display the epoch on screen.
                       text2=str("Avg.: %f" % round(average,6)),
                       text3=str("Avg. (last 20): %f" % round(fiftyaverage,3)),
                       text4=str("High score: %g" % learner.high_score()),
                       tick_length=1,          # Make game ticks super fast.
                       action_callback=learner.action_callback,
                       reward_callback=learner.reward_callback)

  # Loop until you hit something.
  while swing.game_loop():
    pass

  learner.eps_incr()
  # Reset the state of the learner.
  scores[i] = learner.getscore()
  learner.reset()



