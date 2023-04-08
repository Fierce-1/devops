import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const colors = ['#539165', '#4C4B16', '#3E54AC', '#332C39', '#AF19FF', '#191825', '#FF1744'];

const BarChartExample = ({ data }) => {
  const [selectedUser, setSelectedUser] = useState(Object.keys(data)[0]);

  const handleUserChange = (event) => {
    setSelectedUser(event.target.value);
  };

  const userData = data[selectedUser].taskCategoryOverall;
  const chartData = userData.category.map((category, index) => ({
    name: category,
    value: userData.total[index]
  }));

  return (
    <div>
      <select value={selectedUser} onChange={handleUserChange}>
        {Object.keys(data).map((user) => (
          <option key={user} value={user}>{user}</option>
        ))}
      </select>
      <BarChart width={500} height={220} data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" tick={{ fill: 'white' }} />
        <YAxis tick={{ fill: 'white' }} />
        <Tooltip />
        {chartData.map((entry, index) => (
          <Bar key={`bar-${index}`} dataKey="value" fill={colors[index]} />
        ))}
      </BarChart>
    </div>
  );
};

export default BarChartExample;
