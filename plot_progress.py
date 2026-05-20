import argparse
import csv
import sys
from pathlib import Path
from time import sleep

DEFAULT_AVERAGE_WINDOW = 1


def parse_args():
	parser = argparse.ArgumentParser(
		description="Plot DQN training progress from a progress CSV."
	)
	parser.add_argument("csv_path", nargs="?", default="training_progress.csv")
	parser.add_argument("--output", default="training_progress.png")
	parser.add_argument("--mode", choices=["train", "eval", "all"], default="train")
	parser.add_argument(
		"--average-window",
		type=int,
		default=DEFAULT_AVERAGE_WINDOW,
		help="number of games to average for score and reward",
	)
	parser.add_argument("--show", action="store_true")
	parser.add_argument(
		"--live",
		action="store_true",
		help="refresh the output plot every 10 seconds",
	)
	return parser.parse_args()


def load_rows(csv_path, mode):
	rows = []

	with open(csv_path, "r", newline="") as file:
		reader = csv.DictReader(file)
		for row in reader:
			if mode != "all" and row["mode"] != mode:
				continue

			rows.append({
				"episode": int(row["episode"]),
				"mode_episode": int(row["mode_episode"]),
				"score": float(row["score"]),
				"reward": float(row["reward"]),
				"epsilon": float(row["epsilon"]),
			})

	return rows


def rolling_average(values, window_size):
	averages = []
	window_sum = 0

	for index, value in enumerate(values):
		window_sum += value

		if index >= window_size:
			window_sum -= values[index - window_size]

		window_length = min(index + 1, window_size)
		averages.append(window_sum / window_length)

	return averages


def plot_rows(rows, mode, output_path, show, average_window):
	if not show:
		import matplotlib
		matplotlib.use("Agg")

	import matplotlib.pyplot as plt

	x_key = "episode" if mode == "all" else "mode_episode"
	x_label = "episode" if mode == "all" else f"{mode} episode"
	x = [row[x_key] for row in rows]
	scores = [row["score"] for row in rows]
	epsilons = [row["epsilon"] for row in rows]
	average_scores = rolling_average(scores, average_window)

	fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
	fig.suptitle(f"DQN progress ({mode})")

	axes[0].plot(x, scores, linewidth=0.6, alpha=0.25, label="score")
	axes[0].plot(x, average_scores, linewidth=1.4, label=f"{average_window}-game avg")
	axes[0].set_ylabel(f"score ({average_window}-game avg)")
	axes[0].grid(True, alpha=0.25)
	axes[0].legend(loc="upper left")

	axes[1].plot(x, epsilons, linewidth=1.2, color="tab:green")
	axes[1].set_ylabel("epsilon")
	axes[1].set_xlabel(x_label)
	axes[1].grid(True, alpha=0.25)

	fig.tight_layout()

	if output_path:
		fig.savefig(output_path, dpi=160)
		print("Saved plot:", output_path)

	if show:
		plt.show()

	plt.close(fig)


def main():
	args = parse_args()
	csv_path = Path(args.csv_path)

	if not csv_path.exists():
		print(f"CSV not found: {csv_path}", file=sys.stderr)
		return 1

	try:
		rows = load_rows(csv_path, args.mode)
	except (OSError, KeyError, ValueError) as error:
		print(f"Could not read progress CSV: {error}", file=sys.stderr)
		return 1

	if not rows:
		print(f"No rows found for mode: {args.mode}", file=sys.stderr)
		return 1
	if args.average_window <= 0:
		print("--average-window must be greater than 0", file=sys.stderr)
		return 1


	plot_rows(rows, args.mode, args.output, args.show, args.average_window)

	if args.live:
		while True:
			sleep(60)
			rows = load_rows(csv_path, args.mode)
			plot_rows(rows, args.mode, args.output, args.show, args.average_window)


	return 0


if __name__ == "__main__":
	raise SystemExit(main())
