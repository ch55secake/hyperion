# hyperion-go 

This package is responsible for the Go parts of hyperion. This includes an application which should run and recursively train models as well 
as executing trades via the trading212 API. We have left behind an implementation of the random forest algorith for the time being. It is now deprecated 
and will be removed in the future. See GitHub issues for more information about what we are going to do. 

## Requirements
- Go 1.23.3 or later

## Installation
```bash
# Clone the repository
git clone https://github.com/ch55secake/hyperion.git

# Change to project directory
cd hyperion/hyperion-go

# Install dependencies
go mod download
```
