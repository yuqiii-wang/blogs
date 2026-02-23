# R3's Corda

## Quick Start

### Prerequisite

* Use `gradle-7.6.4` (higher versions have conflicts in gradle properties in corda)

### Run 

`cd samples\trader-demo` (this is)
 
`./gradlew samples:trader-demo:deployNodes`

### Config

Originally, corda uses `h2` as DB under `node\src\main\resources\corda-reference.conf`.
It can be replaced with more production-ready DB such as postgresql.
