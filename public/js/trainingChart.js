// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

const drawChart = (chartId, epochs, iter, chartType, values) => {
  // chart configure
  var ctx = document.getElementById(chartId);
  //var epochs = data.config.hparam.epochs;
  var epochs = epochs;
  var labels = [];

  for (var i = 0; i < epochs; i ++){
    labels.push("epoch "+ i);
  }

  var border = chartType == "acc" ?  "rgba(2,117,216,1)" : "rgba(244, 67, 54,1)";
  var background = chartType == "acc" ? "rgba(2,117,216,0.2)" : "rgba(255, 205, 210,0.2)";

  // draw chart
  var chart = new Chart (ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: iter + " " + chartType,
        lineTension: 0.3,
        backgroundColor: background,
        borderColor: border,
        pointRadius: 5,
        pointBackgroundColor: border,
        pointBorderColor: "rgba(255,255,255,0.8)",
        pointHoverRadius: 5,
        pointHoverBackgroundColor: border,
        pointHitRadius: 50,
        pointBorderWidth: 2,
        data: values,
      }],
    },
    options: {
      scales: {
        xAxes: [{
          time: {
            unit: 'date'
          },
          gridLines: {
            display: false
          },
          ticks: {
            maxTicksLimit: 10
          }
        }],
        yAxes: [{
          ticks: {
            // min: 0,
            // max: 30,
            maxTicksLimit: 7
          },
          gridLines: {
            color: "rgba(0, 0, 0, .125)",
          }
        }],
      },
      legend: {
        display: true
      }
    }
  });

  return chart;
}

function updateChart(chart, values) {
  console.log(values);
  chart.data.datasets[0].data = values;
  chart.update();
};