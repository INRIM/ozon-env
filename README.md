<h2 align="center">ozon-env</h2>

<p align="center">
<a href="https://github.com/archetipo/ozon-env"><img alt="Actions Status" src="https://github.com/archetipo/ozon-env/workflows/ci/badge.svg"></a>
<a href="https://coveralls.io/github/archetipo/ozon-env?branch=main"><img alt="Coverage Status" src="https://coveralls.io/repos/github/archetipo/ozon-env/badge.svg?branch=main"></a>
<a href="https://github.com/archetipo/ozon-env/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/archetipo/ozon-env"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# ozon-env

**ozon-env is a runtime self-compiling domain engine.**

It dynamically compiles schema definitions stored in the database into executable domain models at runtime — without requiring application restarts.

Designed to power:

- Web applications  
- Distributed business logic workers  
- Event-driven task processors  
- AI-driven domain agents  

---

## Core Concept

Schema definitions are stored in the database.

At runtime, ozon-env:

1. Reads schema metadata  
2. Generates Python domain models  
3. Dynamically imports and loads them  
4. Executes business logic on top of them

<pre>
Schema (DB)
    ↓
Runtime Model Compilation
    ↓
Domain Model (Pydantic)
    ↓
Worker / Web App / Agent Layer
</pre>

It integrates with the Service App￼ project.

For information about the Service App project,
see https://github.com/INRIM/service-app

Models are regenerated automatically when their schema version changes.

No service restart is required.

---

## Architecture

### Dynamic Model Compilation
- Generates Python models from stored schema
- Uses Pydantic for validation and typing
- Hot-reloads models when schema updates

### Domain Runtime Environment (Env)
- Isolated execution scope
- Session-based lifecycle
- Supports concurrent environments

### Business Logic Workers
- Designed for distributed execution
- Compatible with task brokers (e.g. Redis streams)
- Idempotent task execution
- Suitable for BPMN-driven workflows

### Domain-Aware Execution
- Selection fields and dynamic options
- Nested models
- Datetime normalization
- Data transformation layer

---

## Integration

ozon-env integrates with the  
[Service App project](https://github.com/INRIM/service-app)

Service App provides:
- Web UI
- Schema management
- Workflow integration

ozon-env provides:
- Domain runtime
- Model compilation
- Business logic execution

---

## Installation

### PyPI

```bash
pip install ozon-env
````
or
```bash
poetry add ozon-env
````

### Source Install (Poetry recommended)
```bash
git clone https://github.com/archetipo/ozon-env.git
cd ozon-env
pip install poetry
poetry install
```

### Running Tests

```
./run_test.sh
```

## License

[MIT](LICENSE)

### Designed by Alessio Gerace

## Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.
