
import React from "react";
import { Radar, RadarChart, PolarGrid, Legend, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

const data = [
  {
    subject: 'Initial Access',
    A: 120,
    B: 110,
    fullMark: 150,
  },
  {
    subject: 'Defense Evasion',
    A: 98,
    B: 130,
    fullMark: 150,
  },
  {
    subject: 'Exfiltration',
    A: 86,
    B: 130,
    fullMark: 150,
  },
  {
    subject: 'Command and Control',
    A: 99,
    B: 100,
    fullMark: 150,
  },
  {
    subject: 'Discovery',
    A: 85,
    B: 90,
    fullMark: 150,
  },
  {
    subject: 'Persistence',
    A: 65,
    B: 125,
    fullMark: 150,
  },
];

export default function DomainChart() {
  return (
    <RadarChart cx={100} cy={100} outerRadius={80} width={200} height={200} data={data}>
      <PolarGrid />
      <PolarAngleAxis dataKey="subject" />
      <PolarRadiusAxis angle={30} domain={[0, 150]} />
      <Radar name="User1" dataKey="A" stroke="#8884d8" fill="#2B84FF" fillOpacity={0.6} />
      <Radar name="User2" dataKey="B" stroke="#82ca9d" fill="#D92A2A" fillOpacity={0.6} />
      <Legend />
    </RadarChart>
  );
}
