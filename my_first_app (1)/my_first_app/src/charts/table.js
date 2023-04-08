import React from "react";
import { useTable } from "react-table";
import '../stylesheet/dashboard.css';

const MyTable = () => {
  const data = [
    { user: "Alice", time: 2, count: 2, pattern: 10 },
    { user: "Bob", time: 13, count: 2, pattern: 8 },
    { user: "Charlie", time: 7, count: 1, pattern: 2 }
  ];

  const columns = React.useMemo(
    () => [
      {
        Header: "User",
        accessor: "user"
      },
      {
        Header: "Time",
        accessor: "time"
      },
      {
        Header: "Count",
        accessor: "count"
      },
      {
        Header: "Pattern",
        accessor: "pattern"
      },
      {
        Header: "Anomaly",
        accessor: (data) => data.time + data.count + data.pattern
      }
    ],
    []
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow
  } = useTable({ columns, data });

  return (
    <table className="anomalytable" {...getTableProps()}>
      <thead>
        {headerGroups.map((headerGroup) => (
          <tr className="tablerow"{...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map((column) => (
              <th className="tablehead" {...column.getHeaderProps()}>{column.render("Header")}</th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody {...getTableBodyProps()}>
        {rows.map((row) => {
          prepareRow(row);
          return (
            <tr className="tablerow"{...row.getRowProps()}>
              {row.cells.map((cell) => {
                return <td className="tabledata"{...cell.getCellProps()}>{cell.render("Cell")}</td>;
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default MyTable;
