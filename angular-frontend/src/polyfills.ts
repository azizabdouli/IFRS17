// src/polyfills.ts

/**
 * 🔧 POLYFILLS POUR APPLICATION IFRS17
 * 
 * Ce fichier inclut les polyfills nécessaires pour supporter
 * l'application Angular sur différents navigateurs et environnements.
 */

/***************************************************************************************************
 * ZONE.JS
 * Zone.js est requis par Angular pour la détection des changements.
 */
import 'zone.js';  // Inclus avec Angular CLI.

/***************************************************************************************************
 * POLYFILLS POUR NAVIGATEURS
 * Polyfills nécessaires pour supporter les navigateurs plus anciens.
 */

/** IE10 et IE11 nécessitent les polyfills suivants. */
// import 'core-js/es6/symbol';
// import 'core-js/es6/object';
// import 'core-js/es6/function';
// import 'core-js/es6/parse-int';
// import 'core-js/es6/parse-float';
// import 'core-js/es6/number';
// import 'core-js/es6/math';
// import 'core-js/es6/string';
// import 'core-js/es6/date';
// import 'core-js/es6/array';
// import 'core-js/es6/regexp';
// import 'core-js/es6/map';
// import 'core-js/es6/weak-map';
// import 'core-js/es6/set';

/** IE10 et IE11 nécessitent ClassList pour les éléments SVG. */
// import 'classlist.js';  // Exécuter `npm install --save classlist.js`.

/**
 * 📊 POLYFILLS POUR CHART.JS ET MÉTRIQUES IFRS17
 * Support amélioré pour les graphiques et visualisations.
 */

/** Polyfill pour Canvas dans les anciens navigateurs */
// import 'core-js/es6/promise';

/**
 * 🌐 POLYFILLS POUR WEB ANIMATIONS
 * Requis pour Angular Animations dans certains navigateurs.
 */
// import 'web-animations-js';  // Exécuter `npm install --save web-animations-js`.

/**
 * 📱 CONFIGURATION POUR APPAREILS MOBILES
 * Configuration spécifique pour l'utilisation sur tablettes et smartphones.
 */

// Support touch events
if ('ontouchstart' in window) {
  console.log('📱 Interface tactile détectée - Configuration mobile activée');
}

/**
 * 🔄 CONFIGURATION DÉVELOPPEMENT
 * Outils de développement pour l'environnement IFRS17.
 */

// Configuration console pour debug
if (typeof window !== 'undefined') {
  (window as any).IFRS17_DEBUG = true;
  console.log('🔧 Mode développement IFRS17 activé');
}

/**
 * 💼 POLYFILLS POUR FONCTIONNALITÉS COMPTABLES
 * Support pour les calculs actuariels et financiers.
 */

// Support pour les calculs de précision financière
// import 'decimal.js';  // Pour éviter les erreurs de calcul JavaScript

// Support pour les dates et formats internationaux
// import 'date-fns';  // Manipulation avancée des dates

/**
 * 🌍 CONFIGURATION INTERNATIONALE
 * Support pour les formats de devises et dates régionales.
 */

// Polyfill pour Intl (si nécessaire)
// import 'core-js/es6/reflect';
// import 'core-js/es7/reflect';

// Configuration locale pour l'Algérie et la France
if (typeof window !== 'undefined' && (window as any).Intl) {
  console.log('🌍 Support international activé pour IFRS17');
}