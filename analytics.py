import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from matplotlib import animation

log_file_path = 'posture_log.json'

# Function to calculate ergonomic score
def calculate_ergonomic_score(posture_log, window_size=86400):  # 24 hours window
    current_time = time.time()
    window_start_time = current_time - window_size
    windowed_log = [entry for entry in posture_log if entry['timestamp'] >= window_start_time]
    if not windowed_log:
        return 100  # No data means no bad posture detected
    bad_posture_count = sum(1 for entry in windowed_log if entry['posture'] == 'bad')
    good_posture_count = len(windowed_log) - bad_posture_count
    score = (good_posture_count / len(windowed_log)) * 100
    return score

# Function to update the plot
def update_plot(frame, fig, ax1, ax2):
    try:
        with open(log_file_path, 'r') as log_file:
            posture_log = json.load(log_file)
    except (FileNotFoundError, json.JSONDecodeError):
        posture_log = []

    timestamps = [entry['timestamp'] for entry in posture_log]
    postures = [entry['posture'] for entry in posture_log]
    ergonomic_score = calculate_ergonomic_score(posture_log)

    # Update the small widget score display
    score_label.config(text=f"Ergonomic Score: {ergonomic_score:.2f}%")

    # Clear the axes
    ax1.clear()
    ax2.clear()

    # Plot the posture counts
    df = pd.DataFrame({'Timestamp': timestamps, 'Posture': postures})
    sns.countplot(x='Posture', data=df, order=['good', 'bad'], palette={'good': '#1E90FF', 'bad': '#FFFFFF'}, ax=ax1)
    ax1.set_title('Posture Counts', fontsize=14, color='white')
    ax1.set_xlabel('Posture', fontsize=12, color='white')
    ax1.set_ylabel('Count', fontsize=12, color='white')
    ax1.set_facecolor('#333333')  # Dark grey background for ax1
    ax1.tick_params(colors='white')

    # Plot the ergonomic score
    labels = ['Good Posture', 'Bad Posture']
    sizes = [ergonomic_score, 100 - ergonomic_score]
    colors = ['#1E90FF', '#FFFFFF']  # Darker blue for good, white for bad
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color': 'white', 'fontsize': 12},
            wedgeprops={'edgecolor': 'black'})  # Add black border to pie chart
    ax2.set_title('Ergonomic Score', fontsize=14, color='white')
    ax2.set_facecolor('#333333')  # Dark grey background for ax2

# Function to reset the posture log
def reset_posture_log():
    with open(log_file_path, 'w') as log_file:
        json.dump([], log_file)
    update_plot(None, fig, ax1, ax2)
    messagebox.showinfo("Reset Score", "Ergonomic score has been reset for the current session.")

# Function to show detailed analytics
def show_analytics():
    analytics_window = tk.Toplevel(root)
    analytics_window.title("Posture Analytics")
    analytics_window.configure(bg='#333333')  # Dark grey background

    # Create a figure and axes for plotting
    global fig, ax1, ax2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # ax1 for bar chart on the left, ax2 for pie chart on the right
    fig.patch.set_facecolor('#333333')  # Match background color
    fig.tight_layout(pad=4.0)
    canvas = FigureCanvasTkAgg(fig, master=analytics_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    reset_button = ttk.Button(analytics_window, text="Reset Score", command=reset_posture_log, style='TButton')
    reset_button.pack(side=tk.BOTTOM, pady=10)
    ani = animation.FuncAnimation(fig, update_plot, fargs=(fig, ax1, ax2), interval=1000, cache_frame_data=False)
    analytics_window.mainloop()

# Initialize the main application window
root = tk.Tk()
root.title("Posture Score Widget")

# Set up the style
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12, 'bold'), foreground='#1E90FF', background='#2196F3', borderwidth=0)  # Blue buttons
style.map('TButton', background=[('active', '#1976D2')])
style.configure('TLabel', font=('Helvetica', 16, 'bold'), background='#333333', foreground='white')

# Create a frame for the top bar with dark grey background
top_frame = tk.Frame(root, bg='#333333', height=30)
top_frame.pack(fill=tk.X, side=tk.TOP)

# Create a frame for the main content with dark grey background
main_frame = tk.Frame(root, bg='#333333')
main_frame.pack(fill=tk.BOTH, expand=True)

# Small widget to display ergonomic score
score_label = ttk.Label(main_frame, text="Ergonomic Score: Calculating...", style='TLabel')
score_label.pack(pady=20)
detail_button = ttk.Button(main_frame, text="Show Details", command=show_analytics, style='TButton')
detail_button.pack(pady=10)

# Configure the window size and appearance
root.geometry('300x150')
root.resizable(False, False)

# Run the application
root.mainloop()
