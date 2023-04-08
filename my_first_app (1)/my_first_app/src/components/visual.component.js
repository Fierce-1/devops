/*import React from 'react';
import { XYPlot, LineSeries, XAxis, YAxis, VerticalGridLines, HorizontalGridLines } from 'react-vis';

const data = [
  { x: 0, y: 8 },
  { x: 1, y: 5 },
  { x: 2, y: 4 },
  { x: 3, y: 9 },
  { x: 4, y: 1 },
  { x: 5, y: 7 },
  { x: 6, y: 6 },
  { x: 7, y: 3 },
  { x: 8, y: 2 },
  { x: 9, y: 0 }
];

const MyChart = () => (
    <XYPlot width={300} height={300}>
      <VerticalGridLines />
      <HorizontalGridLines />
      <XAxis />
      <YAxis />
      <LineSeries data={data} />
    </XYPlot>
  );
  export default MyChart;  

  import React from 'react';
import { RadialChart } from 'react-vis';

const data = [
  { angle: 12, label: '12' },
  { angle: 1, label: '1' },
  { angle: 2, label: '2' },
  { angle: 4, label: '4' },
  { angle: 5, label: '5' },
  { angle: 6, label: '6' },
  { angle: 7, label: '7' },
  { angle: 8, label: '8' },
  { angle: 9, label: '9' },
  { angle: 13, label: '13' },
  { angle: 14, label: '14' },
  { angle: 15, label: '15' },
];

function PieChart() {
  return (
    <RadialChart
      data={data}
      width={400}
      height={400}
      showLabels
    />
  );
}

export default PieChart;



import React from 'react';
import { RadialChart } from 'react-vis';
import { groupBy } from 'lodash';

const data = [12,12,1,2,4,5,6,7,7,7,7,7,7,7,7,7,7,8,8,8,12,12,12,12,13,13,13,14,15,9];

// Count the number of occurrences of each unique value in the data array
const counts = groupBy(data, (d) => d);
const chartData = Object.entries(counts).map(([label, data]) => ({
  label,
  angle: data.length,
}));

function PieChart() {
  return (
    <RadialChart
      data={chartData}
      width={400}
      height={400}
      colorRange={['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']}
      showLabels
    />
  );
}

export default PieChart;


import React from 'react';
import { VictoryPie } from 'victory';

const data = [
  { x: 'A', y: 10 },
  { x: 'B', y: 20 },
  { x: 'C', y: 30 },
  { x: 'D', y: 40 },
];

const colors = ['#FFB6C1', '#87CEFA', '#90EE90', '#FFA07A'];

const MyPieChart = () => {
  return (
    <VictoryPie
      data={data}
      colorScale={colors}
      innerRadius={20}
      padding={10}
    />
  );
};

export default MyPieChart;


import React from 'react';
import { PieChart, Pie, Legend, Tooltip } from 'recharts';

const data = [
  { name: 'A', value: 10 },
  { name: 'B', value: 20 },
  { name: 'C', value: 30 },
  { name: 'D', value: 40 },
];

const colors = ['#FFB6C1', '#87CEFA', '#90EE90', '#FFA07A'];

const MyPieChart = () => {
  return (
    <PieChart width={500} height={500}>
      <Pie
        data={data}
        dataKey="value"
        nameKey="name"
        cx="50%"
        cy="50%"
        outerRadius={150}
        fill="#8884d8"
        label
      />
      <Legend />
      <Tooltip />
    </PieChart>
  );
};

export default MyPieChart;













import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend, Tooltip } from 'recharts';

const Graph2 = ({ url, title }) => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const result = await axios.get(url);
      setData(JSON.parse(result.data));
    };

    fetchData();
  }, [url]);

  let dataKey = ""
    dataKey = Object.keys(data[0])[0]
    dataKey1 = Object.keys(data[0])[1]
  }

  return (
    <div style={{ backgroundColor: '#05051C', width: '50%', padding: '10px' }}>
      <h1 style={{ color: '#602BF8', textAlign: 'center', marginBottom: '20px' }}>{title}</h1>
      <BarChart width={500} height={300} data={data} style={{ backgroundColor: '#1C2555', borderRadius: '5px', boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)' }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
        <XAxis dataKey={dataKey} stroke="#fff" />
        <YAxis stroke="#fff" />
        <Tooltip cursor={{ fill: 'rgba(0, 0, 0, 0.2)' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} />
        <Bar dataKey={dataKey1} fill='#87CEFA' />
      </BarChart>
    </div>
  );
};

const Graph = () => {
  const dataArr = [
    { id: 1, url: 'http://localhost:5000/plot', title: 'User Anomalies'},
    { id: 2, url: 'http://localhost:5000/plot1', title: 'Total Events'},
    { id: 3, url: 'http://localhost:5000/plot2', title: 'Total Events per Day'},

  ];

  const rows = [];
  let row = [];

  for (let i = 0; i < dataArr.length; i++) {
    row.push(
      <Graph2 key={dataArr[i].id} url={dataArr[i].url} title={dataArr[i].title} />
    );

    if ((i + 1) % 2 === 0 || i === dataArr.length - 1) {
      rows.push(
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }} key={i}>
          {row}
        </div>
      );
      row = [];
    }
  }

  return (
    <div style={{ backgroundColor: '#05051C', minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h1 style={{ color: '#602BF8', textAlign: 'center', margin: '20px 0' }}>Graphs</h1>
      {rows}
    </div>
  );
};

export default Graph;


import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell } from 'recharts';

const data = [
  { name: ['A','B','C','D'], value: [10,11,22,33] },
 
];

const colors = ['#FFB6C1', '#87CEFA', '#90EE90', '#FFA07A'];

const MyBarChart = () => {
  return (
    <BarChart width={500} height={500} data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Legend />
      <Bar dataKey="value" fill={colors}>
        {
          data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))
        }
      </Bar>
    </BarChart>
  );
};

export default MyBarChart;



import React from 'react';
import { PieChart, Pie, Cell, Legend, Tooltip } from 'recharts';

const COLORS = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'];

const PieChartComponent = ({ chartData }) => {

  if (!chartData) {
    return null;
  }
  
  const chartDataArray = chartData.date.map((date, index) => ({
    name: date,
    value: chartData.total[index],
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active) {
      return (
        <div style={{ backgroundColor: '#fff', padding: '10px', border: '1px solid #ccc' }}>
          <p>{`${label}: ${payload[0].value}`}</p>
        </div>
      );
    }

    return null;
  };

  return (
    <PieChart width={400} height={400}>
  <Pie data={chartDataArray} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} innerRadius={40} fill="#8884d8">
    {chartDataArray.map((entry, index) => (
      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
    ))}
  </Pie>
  <Tooltip />
      <Legend layout="vertical" align="right" verticalAlign="middle" />
</PieChart>

  );
};

export default PieChartComponent;





import React, { useState } from 'react';
import Plot from 'react-plotly.js';

function PythonChart() {
const [selectedUser, setSelectedUser] = useState(null);

const handleUserChange = (event) => {
setSelectedUser(event.target.value);
};

// Define plot data for each user
const plotDataByUser = {
'User1': {
data: [{ x: [1, 2, 3], y: [2, 3, 1], type: 'scatter', mode: 'lines+markers', marker: { color: 'red' } }],
layout: { title: 'Graph for User1' }
},
'User2': {
data: [{ x: [1, 2, 3], y: [3, 1, 2], type: 'scatter', mode: 'lines+markers', marker: { color: 'blue' } }],
layout: { title: 'Graph for User2' }
},
'User3': {
data: [{ x: [1, 2, 3], y: [1, 2, 3], type: 'scatter', mode: 'lines+markers', marker: { color: 'green' } }],
layout: { title: 'Graph for User3' }
}
};

const plotData = plotDataByUser[selectedUser];

return (
<div>
<label htmlFor="user-select">Select user:</label>
<select id="user-select" onChange={handleUserChange}>
<option value="">-- Select User --</option>
<option value="User1">User1</option>
<option value="User2">User2</option>
<option value="User3">User3</option>
</select>
{plotData && <Plot data={plotData.data} layout={plotData.layout} />}
</div>
);
}

export default PythonChart;

*/


import React from 'react';
import Chart from './checkchart.component';

function App() {
  const data = {
    'Ali': {
      'eventperday': {
        'date': ['2016-08-25' ,'2016-08-26','2016-08-27','2016-08-28'],
        'total': [40, 208, 221, 156]
      }
    },
    'Ehtisham': {
      'eventperday': {
        'date': ['2016-08-25' ,'2016-08-26','2016-08-27','2016-08-28'],
        'total': [40, 208, 22, 56]
      }
    }
  };

  return (
    <div>
      <Chart data={data} />
    </div>
  );
}

export default App;