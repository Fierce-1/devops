import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import '../stylesheet/dashboard.css';

function Chart({ data }) {
  const [selectedUser, setSelectedUser] = useState(Object.keys(data)[0]);
  const svgRef = useRef();

  useEffect(() => {
    const userData = data[selectedUser].eventperday;
    const chartData = userData.date.map((date, index) => ({
      event: date,
      total: userData.total[index]
    }));

    const width = 150;
    const height = 150;
    const radius = Math.min(width, height) / 2;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);

      const color = d3.scaleOrdinal()
      .range(['#D27685', '#8D7B68', '#37306B', '#191825', '#66347F']);

    const pie = d3.pie()
      .sort(null)
      .value(d => d.total);

    const data_ready = pie(chartData);

    g.selectAll('path')
      .data(data_ready)
      .enter()
      .append('path')
      .attr('d', d3.arc()
        .innerRadius(0)
        .outerRadius(radius)
      )
      .attr('fill', d => color(d.data.event))
      .attr('stroke', 'white')
      .style('stroke-width', '2px')
      .style('opacity', 1)
      .on('mouseover', function(event, d) {
        d3.select(this).style('opacity', 0.7);
        const total = d.data.total;
        const eventDate = d.data.event;
        const tooltip = d3.select('#tooltip')
          .style('visibility', 'visible')
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 25) + 'px');
        tooltip.html(`<div>Date: ${eventDate}</div><div>Total: ${total}</div>`);
      })
      .on('mouseout', function(event, d) {
        d3.select(this).style('opacity', 1);
        d3.select('#tooltip').style('visibility', 'hidden');
      });

  }, [selectedUser, data]);

  function handleUserSelection(e) {
    setSelectedUser(e.target.value);
  }

  const dropdownOptions = Object.keys(data).map((user) => (
    <option key={user} value={user}>{user}</option>
  ));

  return (
    <div className='piechartbox'>
      <div>
        <label htmlFor="user-dropdown" className='changeColor'>Select User:</label>
        <select id="user-dropdown" value={selectedUser} onChange={handleUserSelection}>
          {dropdownOptions}
        </select>
      </div>
      <div ref={svgRef}></div>
      <div id="tooltip" className='changeColor'></div>
    </div>
  );
}

export default Chart;