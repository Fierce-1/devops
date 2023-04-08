import React from 'react';
import GaugeChart from 'react-gauge-chart';

const GaugeChartExample = ({value = 0.66}) => {
  return (
    <GaugeChart
      id="gauge-chart1" // unique ID for this chart
      percent={value} // percent value
      arcWidth={0.25} // width of the gauge arc
      textColor={'#666'} // color of the text label
      colors={['#332C39', '#FFC371', '#36A2EB']}
    />

  );
};

const GaugeChartExample2 = ({value = 0.66}) => {
    return (
        <GaugeChart id="gauge-chart2" 
        nrOfLevels={10} 
        arcPadding={0.1} 
        cornerRadius={3} 
        percent={value} 
      />
  
    );
  };

  const GaugeChartExample3 = ({value = 0.66}) => {
    return (
        <GaugeChart id="gauge-chart5"
            nrOfLevels={420}
            arcsLength={[0.3, 0.5, 0.2]}
            colors={['#A84448', '#F5CD19', '#EA4228']}
            percent={value}
            arcPadding={0.02}
        />
  
    );
  };


export {GaugeChartExample, GaugeChartExample2, GaugeChartExample3};
