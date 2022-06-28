"""Progress bar"""
import os
import sys
import time


def format_time(seconds):
    """Format time to days:hh:mm:ss:micro"""
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    time_string = ""
    i = 1
    if days > 0:
        time_string += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        time_string += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        time_string += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        time_string += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        time_string += str(millis) + "ms"
        i += 1
    if time_string == "":
        time_string = "0ms"
    return time_string


def progress_bar(current, total, msg=None):
    """Progress_bar: a progress bar mimicking xlua.progress."""
    _, term_width = os.popen("stty size", "r").read().split()
    term_width = int(term_width)

    total_bar_length = 65.0
    last_time = time.time()
    begin_time = last_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(total_bar_length * current / total)
    rest_len = int(total_bar_length - cur_len) - 1

    sys.stdout.write(" [")
    for _ in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for _ in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    tot_time = cur_time - begin_time

    list_of_timings = []
    list_of_timings.append(f"  Step: {format_time(step_time)}")
    list_of_timings.append(f" | Tot: {format_time(tot_time)}" )
    if msg:
        list_of_timings.append(" | " + msg)

    msg = "".join(list_of_timings)
    sys.stdout.write(msg)
    for _ in range(term_width - int(total_bar_length) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for _ in range(term_width - int(total_bar_length / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(f" {current + 1}/{total} ")

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()
