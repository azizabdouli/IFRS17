// src/environments/environment.ts

export const environment = {
  production: false,
  apiUrl: 'http://127.0.0.1:8001',
  apiVersion: 'v1',
  enableLogging: true,
  features: {
    enableAI: true,
    enableML: true,
    enableRealTimeUpdates: true,
    enableAdvancedAnalytics: true
  },
  cache: {
    defaultTtl: 60000, // 1 minute en dev
    maxSize: 50
  },
  ui: {
    refreshInterval: 10000, // 10 secondes en dev
    enableAnimations: true,
    theme: 'blue-light'
  }
};