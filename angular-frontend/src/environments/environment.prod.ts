// src/environments/environment.prod.ts

export const environment = {
  production: true,
  apiUrl: 'http://127.0.0.1:8001',
  apiVersion: 'v1',
  enableLogging: false,
  features: {
    enableAI: true,
    enableML: true,
    enableRealTimeUpdates: true,
    enableAdvancedAnalytics: true
  },
  cache: {
    defaultTtl: 300000, // 5 minutes
    maxSize: 100
  },
  ui: {
    refreshInterval: 30000, // 30 secondes
    enableAnimations: true,
    theme: 'blue-light'
  }
};