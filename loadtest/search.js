// Prueba de carga de la API pública de búsqueda (k6). No corre en CI: se lanza a mano contra
// local o producción (ver README.md de esta carpeta).
//
//   BASE_URL   raíz del servicio (por defecto http://localhost:8002)
//   API_KEY    una API key de PRUEBA cuyo usuario tenga una lista pública ya entrenada
//   LIST_NAME  nombre de esa lista (por defecto "Productos")
//   THINK_S    pausa entre peticiones de cada VU, en segundos (por defecto 0.1)
//
// Escenario: 20 usuarios virtuales durante 30 s sobre POST /api/v1/search, rotando términos
// realistas. Umbrales: p95 < 500 ms y menos de un 1 % de peticiones fallidas (cualquier
// estado ≠ 2xx cuenta como fallo; los 429 se separan en la métrica `rate_limited` para
// distinguir "el servicio va lento" de "el cupo del usuario/IP es demasiado bajo").

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8002';
const API_KEY = __ENV.API_KEY;
const LIST_NAME = __ENV.LIST_NAME || 'Productos';
const THINK_S = Number(__ENV.THINK_S || '0.1');

const TERMS = [
  'cámara de fotos',
  'auriculares sin cables',
  'teléfono móvil barato',
  'portátil para estudiar',
  'altavoz bluetooth',
  'reloj inteligente',
  'martillo',
  'tornillos de madera',
  'camara reflex',
  'monitor 27 pulgadas',
];

const rateLimited = new Rate('rate_limited');
const degraded = new Rate('degraded');
const serviceDuration = new Trend('service_duration_ms', true); // `duration_ms` que reporta el servicio

export const options = {
  scenarios: {
    search: {
      executor: 'constant-vus',
      vus: 20,
      duration: '30s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export function setup() {
  if (!API_KEY) {
    throw new Error('Falta API_KEY (una API key de prueba con una lista pública entrenada)');
  }
  const ready = http.get(`${BASE_URL}/ready`);
  if (ready.status !== 200) {
    throw new Error(`${BASE_URL}/ready respondió ${ready.status}: el servicio no está listo`);
  }
}

export default function () {
  const term = TERMS[Math.floor(Math.random() * TERMS.length)];
  const res = http.post(
    `${BASE_URL}/api/v1/search`,
    JSON.stringify({ list_name: LIST_NAME, search_term: term, limit: 20 }),
    {
      headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
      tags: { name: 'POST /api/v1/search' },
    },
  );

  rateLimited.add(res.status === 429);
  const ok = check(res, {
    'status 200': (r) => r.status === 200,
    'success: true': (r) => r.status === 200 && r.json('success') === true,
    'tiene resultados': (r) => r.status === 200 && r.json('total_results') > 0,
  });
  if (ok) {
    degraded.add(res.json('degraded') === true);
    serviceDuration.add(res.json('duration_ms'));
  }

  if (THINK_S > 0) {
    sleep(THINK_S);
  }
}
