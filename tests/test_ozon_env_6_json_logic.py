from ozonenv.core.BaseModels import CoreModel
from test_common import *
from test_ozon_env_5_basic_for_worker import MockWorker1

pytestmark = pytest.mark.asyncio


class MockWorker3(MockWorker1):

    async def process_document(self, data_doc) -> CoreModel:
        query = {
            "$and": [
                {"active": True},
                {"document_type": {"$in": ["ordine"]}},
                {"numeroRegistrazione": 9},
                {"annoRif": 2022},
            ]
        }
        doc = await self.p_model.load(query)
        assert self.p_model.model.conditional() == {}
        assert self.p_model.model.select_fields() == {
            'document_type': {
                'default': '',
                'multi': False,
                'properties': {},
                'resource_id': '',
                'src': 'values',
            },
            'partner': {
                'default': '',
                'header_key': '',
                'header_value_key': '',
                'multi': False,
                'properties': {
                    'domain': '{}',
                    'id': 'rec_name',
                    'label': 'title',
                    'model': 'posizione',
                },
                'src': 'url',
                'url': '/models/distinct',
            },
            'stato': {
                'default': 'caricato',
                'multi': False,
                'properties': {'readonly': 'si'},
                'resource_id': '',
                'src': 'values',
            },
            'tipi_dettaglio': {
                'default': '',
                'multi': True,
                'properties': {'readonly': 'y'},
                'resource_id': '',
                'src': 'values',
            },
        }

        assert self.p_model.model.select_options() == {
            'document_type': [
                {'label': 'Ordine', 'value': 'ordine'},
                {'label': 'Fattura', 'value': 'fattura'},
                {'label': 'Commessa', 'value': 'commessa'},
                {'label': 'Rda', 'value': 'rda'},
                {
                    'label': 'Rda fondo economale',
                    'value': 'rda_fondo_economale',
                },
                {'label': 'Reso', 'value': 'reso'},
            ],
            'partner': {},
            'stato': [
                {'label': 'Caricato', 'value': 'caricato'},
                {'label': 'In Corso', 'value': 'aperto'},
                {'label': 'Parziale', 'value': 'parziale'},
                {'label': 'Completato', 'value': 'completato'},
                {'label': 'Annullato', 'value': 'annullato'},
            ],
            'tipi_dettaglio': [
                {'label': 'Bene', 'value': 'bene'},
                {'label': 'Consumabile', 'value': 'consumabile'},
                {'label': 'Servizio', 'value': 'servizio'},
            ],
        }
        assert self.p_model.model.logic() == {
            'active': [
                {
                    'actions': [
                        {
                            'name': 'display field',
                            'property': {
                                'label': 'Hidden',
                                'type': 'boolean',
                                'value': 'hidden',
                            },
                            'state': False,
                            'type': 'property',
                        }
                    ],
                    'name': 'chk user',
                    'trigger': {
                        'json': {'var': 'form.is_admin'},
                        'type': 'json',
                    },
                }
            ]
        }
        self.p_model.model.select_options("tipi_dettaglio",  [
                {'label': 'Bene', 'value': 'bene'},
                {'label': 'Consumabile', 'value': 'consumabile'},
                {'label': 'Servizio', 'value': 'servizio'},
                {'label': 'Conto Terzi', 'value': 'conto_terzi'},
            ])
        assert len(self.p_model.model.select_options("tipi_dettaglio")) == 4
        return doc


@pytestmark
async def test_check_logic():
    worker = MockWorker3()
    res = await worker.make_app_session(
        use_cache=True,
        redis_url="redis://localhost:10001",
        params={
            "current_session_token": "BA6BA930",
            "topic_name": "test_topic",
            "document_type": "standard",
            "model": "documento_beni_servizi",
            "session_is_api": False,
            "action_next_page": {
                "success": {"form": "/open/doc"},
            },
        },
    )
    assert res.fail is False
    assert res.data['test_topic']["error"] is False
    assert res.data['test_topic']["done"] is True
    assert res.data['test_topic']['next_page'] == "/open/doc/DOC99998"
    assert res.data['test_topic']['model'] == "documento_beni_servizi"
    assert res.data['documento_beni_servizi']['stato'] == "caricato"
