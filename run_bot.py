import argparse
import haxball.async_common as async_common

from bots.bot_example import ChasingBot
from bots.my_model_bot import MyModelBot

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--channel", type=int, default=0, help="Different agents must be placed on different communication channels.")
	args = parser.parse_args()
	try:
		async_common.run(MyModelBot(str(args.channel)).play())
	except KeyboardInterrupt:
		pass