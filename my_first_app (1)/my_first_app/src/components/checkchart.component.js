/*import React, { useState, useEffect } from 'react';
import * as d3 from 'd3';

function Chart({ data }) {
  const [selectedUser, setSelectedUser] = useState(Object.keys(data)[0]);

  useEffect(() => {
    const userData = data[selectedUser].eventperday;
    const chartData = {
      labels: userData.date,
      values: userData.total,
    };

    d3.select('#line-chart').selectAll('*').remove();

    const width = 400;
    const height = 400;

    const margin = {
      top: 20,
      right: 20,
      bottom: 30,
      left: 50,
    };

    const svg = d3.select('#line-chart')
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const xScale = d3.scaleBand()
      .range([margin.left, width - margin.right])
      .domain(chartData.labels)
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .range([height - margin.bottom, margin.top])
      .domain([0, d3.max(chartData.values)]);

    const xAxis = d3.axisBottom(xScale);

    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickSizeInner(-width + margin.left + margin.right)
      .tickSizeOuter(0);

    svg.append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(xAxis);

    svg.append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(yAxis);

    const line = d3.line()
      .x((d, i) => xScale(chartData.labels[i]))
      .y(d => yScale(d));

    svg.append('path')
      .datum(chartData.values)
      .attr('fill', 'none')
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 1.5)
      .attr('d', line);

  }, [selectedUser, data]);

  function handleUserSelection(e) {
    setSelectedUser(e.target.value);
  }

  const dropdownOptions = Object.keys(data).map((user) => (
    <option key={user} value={user}>{user}</option>
  ));

  return (
    <div>
      <div>
        <label htmlFor="user-dropdown">Select User:</label>
        <select id="user-dropdown" value={selectedUser} onChange={handleUserSelection}>
          {dropdownOptions}
        </select>
      </div>
      <div id="line-chart"></div>
    </div>
  );
}

export default Chart;




import React, { useState, useEffect } from 'react';
import * as d3 from 'd3';
import '../stylesheet/Chart.css'; // import external CSS file

function Chart({ data }) {
  const [selectedUser, setSelectedUser] = useState(Object.keys(data)[0]);

  useEffect(() => {
    const userData = data[selectedUser].eventperday;
    const chartData = {
      labels: userData.date,
      values: userData.total,
    };

    d3.select('#line-chart').selectAll('*').remove();

    const width = 400;
    const height = 400;

    const margin = {
      top: 20,
      right: 20,
      bottom: 30,
      left: 50,
    };

    const svg = d3.select('#line-chart')
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const xScale = d3.scaleBand()
      .range([margin.left, width - margin.right])
      .domain(chartData.labels)
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .range([height - margin.bottom, margin.top])
      .domain([0, d3.max(chartData.values)]);

    const xAxis = d3.axisBottom(xScale);

    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickSizeInner(-width + margin.left + margin.right)
      .tickSizeOuter(0);

    svg.append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(xAxis);

    svg.append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(yAxis);

    const line = d3.line()
      .x((d, i) => xScale(chartData.labels[i]))
      .y(d => yScale(d));

    svg.append('path')
      .datum(chartData.values)
      .attr('fill', 'none')
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 1.5)
      .attr('d', line);

  }, [selectedUser, data]);

  function handleUserSelection(e) {
    setSelectedUser(e.target.value);
  }

  const dropdownOptions = Object.keys(data).map((user) => (
    <option key={user} value={user}>{user}</option>
  ));

  return (
    <div className="chart-container">
      <div>
        <label htmlFor="user-dropdown" className="chart-label">Select User:</label>
        <select id="user-dropdown" className="chart-dropdown" value={selectedUser} onChange={handleUserSelection}>
          {dropdownOptions}
        </select>
      </div>
      <div id="line-chart" className="chart"></div>
    </div>
  );
}

export default Chart;
*/



import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

function Chart({ data }) {
  const [selectedUser, setSelectedUser] = useState(Object.keys(data)[0]);
  const svgRef = useRef();

  useEffect(() => {
    const userData = data[selectedUser].eventperday;
    const chartData = userData.date.map((date, index) => ({
      event: date,
      total: userData.total[index]
    }));

    const width = 200;
    const height = 200;
    const radius = Math.min(width, height) / 2;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);

    const color = d3.scaleOrdinal()
      .range(d3.schemeCategory10);

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
    <div>
      <div>
        <label htmlFor="user-dropdown">Select User:</label>
        <select id="user-dropdown" value={selectedUser} onChange={handleUserSelection}>
          {dropdownOptions}
        </select>
      </div>
      <div ref={svgRef}></div>
      <div id="tooltip"></div>
    </div>
  );
}

export default Chart;