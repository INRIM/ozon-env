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
- JWT-authenticated user lifecycle
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
export OZON_KEYCLOAK_JWKS_URL=https://keycloak.example/realms/demo/protocol/openid-connect/certs
export OZON_KEYCLOAK_ISSUER=https://keycloak.example/realms/demo
export OZON_OAUTH_URL=https://keycloak.example/realms/demo/protocol/openid-connect/token
export OZON_CLIENT_ID=...
export OZON_CLIENT_SECRET=...
export OZON_TOKEN_AUDIENCE=...
export MODELS_FOLDER=/models
```

In this mode:

- schemas are discovered from MongoDB
- models are generated and cached in `MODELS_FOLDER`
- env activation requires a valid Keycloak JWT
- if the JWT is expired and a `refresh_token` is available, ozon-env refreshes
  it through the OAuth token endpoint
- the `user` collection stores `token` as a dictionary with the current token
  data
- `jobcontext` is a DB model and must be managed in MongoDB like the other
  protected runtime models

### REST mode

```bash
export OZON_BACKEND_INTERFACE=rest
export OZON_REST_BASE_URL=http://base_usr
export OZON_REST_API_PREFIX=/base_usr/v2
export OZON_OAUTH_URL=https://keycloak.example/realms/demo/protocol/openid-connect/token
export OZON_CLIENT_ID=...
export OZON_CLIENT_SECRET=...
export OZON_TOKEN_AUDIENCE=...
export MODELS_FOLDER=/models
```

In this mode:

- `new()` still creates local Python objects
- ORM operations such as `find`, `load`, `insert`, `update`, `upsert`,
  `remove`, `count`, `distinct` are mapped to `POST` operations
- env activation requires a `job_token`; the REST client does not resolve
  `JobContext` locally and does not read the DB
- the REST client can use a configured `rest_token` or generate a dedicated
  M2M token with OAuth `client_credentials`
- every protected API call must also send a `job_token` header
- the REST client never creates or updates `jobcontext`; it only consumes the
  `job_token` issued by the DB-side flow
- `job_token` is validated server-side against the persisted `jobcontext`
  record; the `client_id` in `JobContext` must match the `client_id` claim of
  the M2M token
- `settings` and `component` remain local bootstrap models; `jobcontext`
  remains authoritative in DB and is only consumed by the REST client

Expected REST path pattern:

```text
POST {OZON_REST_BASE_URL}/base_usr/v2/{operation_name}
```

REST API specification:

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `{OZON_REST_API_PREFIX}/{operation_name}` | Executes an ORM operation on the REST backend. |
| `GET` | `{OZON_REST_API_PREFIX}/collections_names` | Returns remote collection names used during bootstrap. |
| `GET` | `{OZON_REST_API_PREFIX}/init_settings/{app_code}` | Returns app settings used during bootstrap. |

Headers:

```http
Authorization: Bearer <token>
job_token: jctx_<generated-token>
Accept: application/json
Content-Type: application/json
```

OAuth token generation:

When no token is already available, `OzonDataApiClient` can generate the M2M
token with OAuth `client_credentials`.

```text
POST {OZON_OAUTH_URL}
Content-Type: application/x-www-form-urlencoded
```

Token request form data:

| Field | Source | Required |
| --- | --- | --- |
| `grant_type` | fixed value `client_credentials` | yes |
| `client_id` | `OZON_CLIENT_ID` or `rest_client_id` / `client_id` config | yes |
| `client_secret` | `OZON_CLIENT_SECRET` or `rest_client_secret` / `client_secret` config | yes |
| `audience` | `OZON_TOKEN_AUDIENCE` or `rest_token_audience` / `token_audience` config | no |

The REST client uses `access_token` from the JSON response as the bearer token.

Standard POST payload:

```json
{
  "model": "user",
  "data_model": "user",
  "domain": {
    "uid": "admin"
  }
}
```

`job_token` is not part of the JSON payload. It must be passed in the HTTP
header together with the bearer token.

### Env Authentication Contract

When an `env` is created, the input token must be a valid Keycloak JWT. The
application entrypoint passes it to `make_app_session()` in
`params["current_token"]`, and ozon-env resolves the authenticated principal in
the `user` collection.

Example:

```python
await env.make_app_session(
    params={
        "current_token": {
            "access_token": "<jwt>",
            "refresh_token": "<refresh-token>",
        }
    }
)
```

Rules:

- `access_token` must be a valid Keycloak JWT
- when `access_token` is expired and `refresh_token` is present, ozon-env
  refreshes it using `grant_type=refresh_token`
- after a successful login or refresh, the token dictionary is persisted on the
  `user` record
- `user_session` is the resolved `User` model; there is no separate runtime
  authentication model in the new flow

### JobContext Security Model

`JobContext` is a persistent security model managed in `db`. It is created by a
user authenticated with a valid JWT and then consumed by `rest` clients.

Model fields:

| Field | Description |
| --- | --- |
| `job_token` | Generated token, for example `jctx_123...` |
| `client_id` | Mandatory client identifier requested by the user |
| `job_key` | Optional input; generated as UUID when omitted |
| `process_instance_key` | Optional input; generated as UUID when omitted |
| `resolved_user_id` | User id resolved from the validated JWT |
| `issued_at` | Creation timestamp |
| `expires_at` | Expiration timestamp |

DB responsibilities:

- full CRUD for the `jobcontext` model
- helper methods `create_job_context()`, `delete_job_context()`,
  `clean_job_contexts()` and `job_done()`
- sidecar validation helpers `validate_job_context()` / `verify_job_context()`
  and `init_api_job_context(m2m_token, job_token)`
- expiration cleanup for invalid or expired contexts

### JobContext Flow

1. A user authenticated with JWT creates a `JobContext` in `db`.
2. ozon-env generates `job_token`, timestamps and default UUID values when
   `job_key` or `process_instance_key` are missing.
3. The REST client calls the remote API with:
   - `Authorization: Bearer <m2m-token>`
   - `job_token: <jctx_...>`
4. The API executes only if:
   - the `job_token` exists and is active
   - the `job_token` is not expired
   - the `client_id` stored in `JobContext` matches the `client_id` carried by
     the M2M token
5. When the job is completed, `job_done()` or `delete_job_context(job_token)`
   removes the `JobContext`.

### Supported POST Operations

| Operation | Required payload fields | Optional payload fields |
| --- | --- | --- |
| `find` | `model`, `data_model`, `domain` | `sort`, `limit`, `skip`, `fields`, `batch_size` |
| `load` | `model`, `data_model`, `domain` | |
| `insert` | `model`, `data_model`, `record` | `is_many` |
| `update` | `model`, `data_model`, `record` | `remove_mata`, `force_update_whole_record` |
| `remove` | `model`, `data_model`, `record` | |
| `remove_all` | `model`, `data_model`, `domain` | |
| `count` | `model`, `data_model`, `domain` | |
| `distinct` | `model`, `data_model`, `field_name`, `query` | |
| `aggregate` | `model`, `data_model`, `domain` | `sort`, `limit`, `skip`, `pipeline_items`, `obfuscate_fields`, `fields`, `batch_size` |
| `search_all_distinct` | `model`, `data_model`, `distinct`, `query` | `compute_label`, `sort`, `limit`, `skip`, `raw_result` |

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
        "rest_token": "<m2m-token>",
        "rest_oauth_url": "https://keycloak.example/realms/demo/protocol/openid-connect/token",
        "rest_client_id": "...",
        "rest_client_secret": "...",
        "rest_token_audience": "ozon-api",
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
)

await env.make_app_session(
    params={
        "job_token": "jctx_123",
        "current_user": {
            "uid": "optional-local-user-metadata"
        },
    }
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
        "rest_token": "<m2m-token>",
    }
)

await worker.make_app_session(
    params={
        "topic_name": "job",
        "model": "user",
        "job_token": "jctx_123",
    },
    local_model={"user": UserModel},
    settings={"rec_name": "demo", "upload_folder": "/uploads"},
)
```

For REST workers:

- a runtime `job_token` is mandatory for protected operations
- a configured `rest_token` or generated M2M token is mandatory
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
