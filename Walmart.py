import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

# sources https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
# https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
# https://stackoverflow.com/questions/18767523/fitting-data-with-numpy
# https://www.geeksforgeeks.org/python-map-function/
# https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels-in-matplotlib
# https://www.kite.com/python/answers/how-to-set-point-size-in-a-matplotlib-plot-in-python

class Walmart:
    def __init__(self):
        self.population = []
        self.profit = []

    def loadData(self, filename):
        # store the data of cities from filename into the library
        raw_data = open(filename, "r")
        list_data = raw_data.readlines()
        raw_data.close()
        # parse list_data, create city objects, add cities to self.cityArray
        for line in list_data:
            dataArray = line.split(',')
            self.population.append(float(dataArray[0]))
            self.profit.append(float(dataArray[1]))

        # model profit as a function of population
        self.predict = np.poly1d(np.polyfit(self.population, self.profit, 1)) # part b
        # self.predict(x) => y

    def plotData(self): # part a
        font = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 15}
        pointSize = 15*np.ones(len(self.population))

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(self.population, self.profit, pointSize, 'r') # scatter of population vs profit

        # grid details
        major_ticks = np.arange(0, 26, 5) # major ticks every 5
        minor_ticks = np.arange(-5, 26, 1) # minor ticks every 1
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2) # line transparency
        ax.grid(which='major', alpha=0.5)
        plt.setp(ax.get_xticklabels(), fontsize=12) # label font size
        plt.setp(ax.get_yticklabels(), fontsize=12)

        # plot details
        plt.ylabel('Profit $ (x10,000)', fontdict=font)
        plt.xlabel('Population (x10,000)', fontdict=font)
        plt.xlim(min(self.population)-1, 25)
        plt.title("Walmart's Locations' Profit vs Population", fontdict=font)
        plt.show()

    def plotData_with_model(self): # part c
        font = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 15}
        x = sorted(self.population)  # sort since population is unordered
        pointSize = 15 * np.ones(len(self.population))

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(self.population, self.profit, pointSize, 'r')  # scatter of population vs profit
        plt.plot(x, self.predict(x), '--k')  # best fit line part b

        # accuracy of the model
        self.accuracy = mae(self.profit, self.predict(self.population)) # part f

        # grid details
        major_ticks = np.arange(0, 26, 5)  # major ticks every 5
        minor_ticks = np.arange(-5, 26, 1)  # minor ticks every 1
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)  # line transparency
        ax.grid(which='major', alpha=0.5)
        plt.setp(ax.get_xticklabels(), fontsize=12)  # label font size
        plt.setp(ax.get_yticklabels(), fontsize=12)

        # plot details
        plt.ylabel('Profit $ (x10,000)', fontdict=font)
        plt.xlabel('Population (x10,000)', fontdict=font)
        plt.xlim(min(self.population) - 1, 25)
        plt.title("Walmart's Locations' Profit vs Population\n(MAE = %.2f)" % self.accuracy, fontdict=font)
        plt.show()

    def future_store_predictions(self): # part e
        future_stores = {}
        future_stores["A"] = self.predict(7.8)
        future_stores["B"] = self.predict(4.4)
        future_stores["C"] = self.predict(4.7)
        future_stores["D"] = self.predict(6.12)
        future_stores["E"] = self.predict(8.55)
        future_stores["F"] = self.predict(6.7)
        future_stores["G"] = self.predict(9.8)
        future_stores["H"] = self.predict(7.01)

        future_stores_ordered = dict(sorted(future_stores.items(), key=lambda item: item[1], reverse=True))
        print("Future stores in descending order of profit:")
        for k in future_stores_ordered:
            print("Store", k, "expects $%.2f (x10,000) of profit" % future_stores_ordered[k])


if __name__ == '__main__':
    W = Walmart()
    W.loadData("walmart_data.txt")
    W.plotData()
    W.plotData_with_model()
    W.future_store_predictions()