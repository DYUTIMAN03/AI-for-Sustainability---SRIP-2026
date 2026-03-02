/* ═══════════════════════════════════════════════════════════
   SRIP 2026 — Main JavaScript
   Scroll animations, nav behavior, metric bars
   ═══════════════════════════════════════════════════════════ */

// ── Navbar scroll effect ──────────────────────────────────
const navbar = document.getElementById('navbar');
let lastScroll = 0;

window.addEventListener('scroll', () => {
  const scrollY = window.scrollY;
  if (scrollY > 50) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
  lastScroll = scrollY;
});

// ── Mobile nav toggle ─────────────────────────────────────
const navToggle = document.getElementById('navToggle');
const navLinks = document.querySelector('.nav-links');

navToggle.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});

// Close mobile nav when link is clicked
navLinks.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => {
    navLinks.classList.remove('open');
  });
});

// ── Intersection Observer for scroll reveals ──────────────
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        // For metric bars, animate the fill
        const fills = entry.target.querySelectorAll('.metric-fill');
        fills.forEach((fill) => {
          fill.style.width = fill.style.width || '0%';
        });
      }
    });
  },
  {
    threshold: 0.15,
    rootMargin: '0px 0px -50px 0px',
  }
);

// Add reveal class to elements that should animate
function initRevealAnimations() {
  // Pipeline cards with staggered delay
  document.querySelectorAll('.pipeline-card').forEach((card, i) => {
    card.style.transitionDelay = `${i * 0.15}s`;
    revealObserver.observe(card);
  });

  // All section headers
  document.querySelectorAll('.section-header').forEach((el) => {
    el.classList.add('reveal');
    revealObserver.observe(el);
  });

  // Content grids
  document.querySelectorAll('.content-grid, .full-width-image').forEach((el) => {
    el.classList.add('reveal');
    revealObserver.observe(el);
  });

  // Model architecture cards
  document.querySelectorAll('.arch-card').forEach((card, i) => {
    card.classList.add('reveal');
    card.style.transitionDelay = `${i * 0.1}s`;
    revealObserver.observe(card);
  });

  // Results panel
  document.querySelectorAll('.results-panel, .viz-grid, .interpretation').forEach((el) => {
    el.classList.add('reveal');
    revealObserver.observe(el);
  });

  // Image cards
  document.querySelectorAll('.image-card').forEach((el) => {
    el.classList.add('reveal');
    revealObserver.observe(el);
  });
}

// ── Metric bar animation ──────────────────────────────────
const metricObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const fills = entry.target.querySelectorAll('.metric-fill');
        fills.forEach((fill) => {
          // The target width is set as inline style in HTML
          const targetWidth = fill.getAttribute('style')?.match(/width:\s*([^;]+)/)?.[1];
          if (targetWidth) {
            // Reset then animate
            fill.style.width = '0%';
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                fill.style.width = targetWidth;
              });
            });
          }
        });
        metricObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.3 }
);

document.querySelectorAll('.metrics-grid').forEach((grid) => {
  metricObserver.observe(grid);
});

// ── Smooth scroll for anchor links ────────────────────────
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener('click', (e) => {
    e.preventDefault();
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      const offset = 80; // navbar height
      const top = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({
        top,
        behavior: 'smooth',
      });
    }
  });
});

// ── Active nav link tracking ──────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navLinksAll = document.querySelectorAll('.nav-link[href^="#"]');

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navLinksAll.forEach((link) => {
          link.style.color = '';
          link.style.background = '';
        });
        const activeLink = document.querySelector(
          `.nav-link[href="#${entry.target.id}"]`
        );
        if (activeLink) {
          activeLink.style.color = 'var(--accent-green)';
          activeLink.style.background = 'var(--accent-green-dim)';
        }
      }
    });
  },
  { threshold: 0.3 }
);

sections.forEach((section) => {
  sectionObserver.observe(section);
});

// ── Initialize ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initRevealAnimations();
});

// If DOM is already loaded
if (document.readyState !== 'loading') {
  initRevealAnimations();
}
