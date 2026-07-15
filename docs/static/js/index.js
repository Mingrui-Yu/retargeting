window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})

// ================================
// Objective Terms: bullet ↔ equation highlighting
// ================================
document.addEventListener('DOMContentLoaded', function () {
  const box = document.getElementById('objective-terms');
  if (!box) return;

  const items = box.querySelectorAll('.obj-item[data-key]');
  const eqs = box.querySelectorAll('.eq-item[data-key]');

  let pinnedKey = null;

  function clearActive() {
    items.forEach(el => el.classList.remove('is-active'));
    eqs.forEach(el => el.classList.remove('is-active'));
  }

  function setActive(key) {
    clearActive();
    const li = box.querySelector('.obj-item[data-key="' + key + '"]');
    const eq = box.querySelector('.eq-item[data-key="' + key + '"]');
    if (li) li.classList.add('is-active');
    if (eq) li && eq.classList.add('is-active');
  }

  items.forEach(item => {
    const key = item.getAttribute('data-key');

    // hover highlight
    item.addEventListener('mouseenter', () => {
      if (pinnedKey) return;
      setActive(key);
    });

    item.addEventListener('mouseleave', () => {
      if (pinnedKey) return;
      clearActive();
    });

    // click to pin
    item.addEventListener('click', () => {
      if (pinnedKey === key) {
        pinnedKey = null;
        clearActive();
      } else {
        pinnedKey = key;
        setActive(key);

        // optional: scroll equation into view
        const eq = box.querySelector('.eq-item[data-key="' + key + '"]');
        if (eq) {
          eq.scrollIntoView({
            block: 'nearest',
            behavior: 'smooth'
          });
        }
      }
    });
  });
});
