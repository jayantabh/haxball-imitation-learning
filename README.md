# Haxball Imitator
<img src="https://www.haxball.com/FFuuq69i/s/haxball-big-min.png">

Solving Haxball using Imitation Learning methods.

## How to run my agents?

### Make the server running.

1. Add the following line to your hosts file (`c:\windows\system32\drivers\etc\hosts` on windows, `/etc/hosts` on linux)
```
127.0.0.1 inter.haxball.com
```

2. Start the server
```bash
python run_server.py
```

### Call up your agent.

0. Edit `run_bot.py` script to instantiate your agent (an example is provided).

1. Start interaction between the agent and the server. The server allows multiple connections, thus you should provide your connection channel (any integer you like, defaults to 0).
```bash
python run_bot.py --channel-id=0
```

2. Open `inter.haxball.com:8080/#channel_id` in your browser. This is the original game with slight modifications to allow sending states and accepting the inputs from the bot. One important limitation -- the game window must be active, otherwise the communication channel will hang.

## How to parse replays?

1. Run raw replays converter. It will preprocess raw replays, put them into a separate folder, and create a file with nickname mapping (nickname -> all replays).
```bash
python run_converter.py --path=replays/
```

2. Use utilities provided in `replay.py` to parse converted replays.
