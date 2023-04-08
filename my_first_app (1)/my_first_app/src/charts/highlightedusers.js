/*import React from 'react';
import ProgressBar from 'react-bootstrap/ProgressBar';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../stylesheet/dashboard.css';

const Highlightedusers = ({data}) => {

  // Create a new array of objects where each object has a name and an anomalies property
  const sortedData = data.name.map((name, index) => ({ name, anomalies: data.anomalies[index] }))
    .sort((a, b) => b.anomalies - a.anomalies);
  
  // Define an array of colors to use for the progress bars
  const colors = ['danger', 'warning', 'info', 'success', 'primary'];

  return (
    <div>
      <h4 className='txtprogressbar'>Highlighted Users</h4>
      {sortedData.slice(0, 7).map((user, index) => (
        <div key={user.name}>
          <h5 className='txtprogressbar'>{user.name}</h5>
          <ProgressBar
            now={user.anomalies}
            label={`${user.anomalies}`}
            variant={colors[index]}
            style={{ minWidth: '30px', maxWidth: '800px', width: '300px'}}
          />
        </div>
      ))}
    </div>
  );
};

export default Highlightedusers;
*/


import React from 'react';
import ProgressBar from 'react-bootstrap/ProgressBar';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../stylesheet/dashboard.css';

const Highlightedusers = ({data}) => {

  // Create a new array of objects where each object has a name and an anomalies property
  const sortedData = data.name.map((name, index) => ({ name, anomalies: data.anomalies[index] }))
    .sort((a, b) => b.anomalies - a.anomalies);
  
  return (
    <div>
      <h4 className='txtprogressbar'>Highlighted Users</h4>
      {sortedData.slice(0, 6).map((user, index) => (
        <div key={user.name}>
          <h5 className='txtprogressbar'>{user.name}</h5>
          <ProgressBar
            now={user.anomalies}
            label={`${user.anomalies}`}
            className='custom-progress'
            style={{ minWidth: '30px', maxWidth: '800px', width: '300px'}}
          />
        </div>
      ))}
    </div>
  );
};

export default Highlightedusers;