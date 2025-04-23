class Replay_Buffer():
    """
        The replay buffer stores the transitions that the agent observes, allowing us to reuse this data later.

        Args:
            env: The environment to interact with
            fullsize: The maximum size of the replay buffer
            minsize: The minimum size of the replay buffer before the agent starts learning
            batchsize: The batch size used for training
    """
    def __init__(self, env, fullsize=BUFFER_SIZE, minsize=MIN_REPLAY_SIZE, batchsize=BATCH_SIZE):
      pass

    def append(self, transition):
      pass

    def sample_batch(self):
      pass
    def initialize(self):
      pass

def iou(bbox1, target_bbox):
  pass
