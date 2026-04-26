<h2 align="center">ozon-env</h2>

<p align="center">
<a href="https://github.com/archetipo/ozon-env"><img alt="Actions Status" src="https://github.com/archetipo/ozon-env/workflows/ci/badge.svg"></a>
<a href="https://coveralls.io/github/archetipo/ozon-env?branch=main"><img alt="Coverage Status" src="https://coveralls.io/repos/github/archetipo/ozon-env/badge.svg?branch=main"></a>
<a href="https://github.com/archetipo/ozon-env/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/archetipo/ozon-env"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# ozon-env

**ozon-env is a runtime self-compiling domain engine.**

It dynamically compiles schema definitions into executable domain models at
runtime, without requiring application restarts.

It can run in two modes:

- `db`: schema and data are loaded from MongoDB
- `rest`: schema is loaded from `components` or from generated models in
  `MODELS_FOLDER`, while ORM operations are mapped to HTTP `POST` calls

Designed to power:

- Web applications  
- Distributed business logic workers  
- Event-driven task processors  
- AI-driven domain agents  

---

## Core Concept

Schema definitions can come from the database or from a local list of
FormIO-like `components`.

At runtime, ozon-env:

1. Reads schema metadata  
2. Generates Python domain models  
3. Dynamically imports and loads them  
4. Executes business logic on top of them

<pre>
Schema (DB or components)
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
- Selectable backend interface: `db` or `rest`

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

## Backend Interface

### Default DB mode

`db` is the default backend.

```bash
export OZON_BACKEND_INTERFACE=db
export MONGO_USER=...
export MONGO_PASS=...
export MONGO_URL=...
export MONGO_DB=...
export MODELS_FOLDER=/models
```

In this mode:

- schemas are discovered from MongoDB
- models are generated and cached in `MODELS_FOLDER`
- session validation is based on the `session` collection

### REST mode

```bash
export OZON_BACKEND_INTERFACE=rest
export OZON_REST_BASE_URL=http://base_usr
export OZON_REST_API_PREFIX=/base_usr/v2
export OZON_REST_TOKEN=...
export MODELS_FOLDER=/models
```

In this mode:

- `new()` still creates local Python objects
- ORM operations such as `find`, `load`, `insert`, `update`, `upsert`,
  `remove`, `count`, `distinct` are mapped to `POST` operations
- default headers use `Authorization: Bearer <token>`
- `settings`, `session` and `component` are handled locally by the env/orm
- session is optional: if no token or local session is provided, the env can
  still run and use the configured `OZON_REST_TOKEN`

Expected REST path pattern:

```text
POST {OZON_REST_BASE_URL}/base_usr/v2/{operation_name}
```

REST bootstrap endpoints used by `OzonOrmRest.init_db_models()`:

```text
GET {OZON_REST_BASE_URL}/base_usr/v2/collections_names
GET {OZON_REST_BASE_URL}/base_usr/v2/init_settings/{app_code}
```

## REST Initialization Example

```python
from ozonenv.OzonEnv import OzonEnvRest

env = OzonEnvRest(
    {
        "app_code": "demo",
        "rest_base_url": "http://base_usr",
        "rest_api_prefix": "/base_usr/v2",
        "rest_token": "token",
        "models_folder": "/tmp/models",
    }
)

await env.init_env(
    components=[...],   # FormIO-like component schemas
    settings={
        "rec_name": "demo",
        "upload_folder": "/uploads",
        "tz": "Europe/Rome",
    },
    sessions=[...],     # optional local session records
)
```

If a generated model already exists in `MODELS_FOLDER`, ozon-env imports it.
If it does not exist, ozon-env generates it from the provided component schema.

If you pass a custom runtime model class to the env, it must inherit from
`OzonModelBase` and expose a coherent `interface_type`:

```python
class MyRestModel(OzonModelRest):
    interface_type = "rest"
```

`OzonEnvBase` validates `cls_model.interface_type` against
`backend_interface` during init. It does not replace `cls_model`.

## Worker Usage in REST Mode

`OzonWorkerEnvRest.make_app_session()` uses the same worker API on REST:

```python
from ozonenv.OzonEnv import OzonWorkerEnvRest

worker = OzonWorkerEnvRest(
    {
        "app_code": "demo",
        "rest_base_url": "http://base_usr",
        "rest_api_prefix": "/base_usr/v2",
        "rest_token": "token",
    }
)

await worker.make_app_session(
    params={"topic_name": "job", "model": "user"},
    local_model={"user": UserModel},
    settings={"rec_name": "demo", "upload_folder": "/uploads"},
)
```

For REST workers:

- a runtime session is optional
- a configured `rest_token` can be enough
- the worker can use the same ORM API used in DB mode
- the actual persistence/query layer is delegated to the REST backend

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
