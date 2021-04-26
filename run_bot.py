import argparse
import haxball.async_common as async_common

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=0,
                        help="Different agents must be placed on different communication channels.")
    parser.add_argument("--bot", type=str, help="Must specify --bot")
    parser.add_argument("--name", type=str, help="Must specify saved model name")
    args = parser.parse_args()
    try:
        if args.bot == "Basic":
            from bots.BasicBot import BasicBot
            async_common.run(BasicBot(str(args.channel), args.name).play())
        elif args.bot == "DistBot":
            from bots.DistBot import DistBot
            async_common.run(DistBot(str(args.channel), args.name).play())
        elif args.bot == "Basic3v3DC":
            from bots.BasicBot3v3DC import BasicBot3v3DC
            async_common.run(BasicBot3v3DC(str(args.channel), args.name).play())
    except KeyboardInterrupt:
        pass
