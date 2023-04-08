import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';

function PythonChart({data}) {
  const [selectedUser, setSelectedUser] = useState(null);

  const handleUserChange = (event) => {
    setSelectedUser(event.target.value);
    console.log('User:',event.target.value)    
  };

  const plotDataByUser = data

  const userOptions = Object.keys(plotDataByUser).map(user => (
    <option value={user} key={user}>{user}</option>
  ));
    console.log('THis is how it comes',plotDataByUser)
    var plotData = plotDataByUser[selectedUser];

  if (selectedUser !== null){
    plotData = plotDataByUser[selectedUser]['name'];
  }
  
   console.log('THis is to be ploted',plotData) 

  return (
    <div>
      <label htmlFor="user-select">Select user:</label>
      <select id="user-select" onChange={handleUserChange}>
      <option value="">-- Please Select --</option>
        {userOptions}
      </select>
      <Line
        data={{
          labels: plotData.x,
          datasets: [{
            label: 'My Graph',
            data: plotData.y,
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
          }],
        }}
      />
    </div>
  );
}

export default PythonChart;
