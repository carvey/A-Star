# 8 Puzzle Problem
Using A\* search algorithm for GSU CSCI 5430

## Overview:
In this project you will implement a program to solve the 8-puzzle problem using the A\*, IDA\* or RBFS algorithm. Your program should
be a command line based program reading input from stdin and print results to stdout.

## Input and output formats:
The input file will contain the initial and goal board configuration.

A sample can look like this:
```
7 2 4
5 0 6
8 3 1

0 1 2
3 4 5
6 7 8
```

## Output:
If the goal configuration is reachable in a finite number of steps, output all board configurations from initial to goal and the minimum number of steps required to reach the goal.

If the goal configuration is not reachable in a finite number of steps, output "no solution".
