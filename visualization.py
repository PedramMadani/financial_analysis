import matplotlib.pyplot as plt
import os


def visualize_data(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['SMA'], label='SMA', color='red', linestyle='--')
    plt.plot(data['EMA'], label='EMA', color='green', linestyle='--')
    plt.title('Stock Price with Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Define the path for saving the plot
    output_folder = 'output'
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    plot_filename = 'stock_price_with_indicators.png'
    plot_path = os.path.join(output_folder, plot_filename)

    plt.savefig(plot_path)  # Save the figure
    plt.close()  # Close the figure to free up memory

    print(f"Plot saved to {plot_path}")
