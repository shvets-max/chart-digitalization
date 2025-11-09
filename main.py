import matplotlib.pyplot as plt

from chart_extraction import extract_time_series

if __name__ == "__main__":
    ts = extract_time_series("data/img_5.png")
    # plot time series
    dates = [t[0] for t in ts if t is not None]
    values = [t[1] for t in ts if t is not None]
    plt.plot(dates, values)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Extracted Time Series")
    plt.show()
