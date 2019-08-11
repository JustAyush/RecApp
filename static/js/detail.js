// send click flag on detail page load

$(document).ready(function () {
  function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
      var cookies = document.cookie.split(';');
      for (var i = 0; i < cookies.length; i++) {
        var cookie = jQuery.trim(cookies[i]);
        // Does this cookie string begin with the name we want?
        if (cookie.substring(0, name.length + 1) == (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  // setTimeout(fetch("detail_url", {
  //   method: "POST", 
  //   credentials: "same-origin",
  //   headers: {
  //       "X-CSRFToken": getCookie("csrftoken"),
  //       "Accept": "application/json",
  //       "Content-Type": "application/json",
  //       'X-Requested-With': 'XMLHttpRequest'
  //   },
  //   body: JSON.stringify({clicked: 'clicked'})
  // }).then(res => {
  //   console.log("Request complete! response:", res);
  // }), 10000);

  setTimeout(function () {
    // console.log(detail_url)
    console.log("oh")
    // console.log(type(detail_url))
    fetch(`/example/detail/${isbn}`, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "X-CSRFToken": getCookie("csrftoken"),
        "Accept": "application/json",
        "Content-Type": "application/json",
        'X-Requested-With': 'XMLHttpRequest'
      },
      body: JSON.stringify({
        clicked: 'clicked',
        ISBN: isbn
      })
    }).then(res => {
      console.log("Request complete! response:", res);
    })

  }, 4000);
});




// For converting number into rating

var djangoData = $('#my-data').data();
console.log(`djangoData: ${djangoData}`);
// Initial Ratings
const ratings = {
  rating: djangoData.name,
}
console.log(`djangoData: ${ratings.rating}`);
// Total Stars
const starsTotal = 5;

// Run getRatings when DOM loads
// document.addEventListener('DOMContentLoaded', getRatings);

// Get ratings

for (let rating in ratings) {
  // Get percentage
  const starPercentage = (ratings[rating] / starsTotal) * 100;

  // Round to nearest 10
  const starPercentageRounded = `${Math.round(starPercentage / 10) * 10}%`;

  console.log(starPercentageRounded);
  // Set width of stars-inner to percentage
  document.querySelector(`.${rating} .stars-inner`).style.width = starPercentageRounded;

  // Add number rating
  // document.querySelector(`.${rating} .number-rating`).innerHTML = ratings[rating];
}



var djangoData1 = $('#my-data1').data();
console.log(`djangoData: ${djangoData}`);
const ratings1 = {
  rating1: djangoData1.name,
}
console.log(`djangoData: ${ratings1.rating1}`);

const starsTotal1 = 5;

for (let rating1 in ratings1) {
  const starPercentage = (ratings1[rating1] / starsTotal1) * 100;

  // Round to nearest 10
  const starPercentageRounded = `${Math.round(starPercentage / 10) * 10}%`;
  console.log(starPercentageRounded);
  // Set width of stars-inner to percentage
  document.querySelector(`.${rating1} .stars-inner1`).style.width = starPercentageRounded;

  // Add number rating
  // document.querySelector(`.${rating} .number-rating`).innerHTML = ratings[rating];
}




// function sendReadIt(){
//   console.log(isbn);
//   fetch("detail_url", {
//     method: "POST", 
//     credentials: "same-origin",
//     headers: {
//         "X-CSRFToken": getCookie("csrftoken"),
//         "Accept": "application/json",
//         "Content-Type": "application/json",
//         'X-Requested-With': 'XMLHttpRequest'
//     },
//     body: JSON.stringify({readIt: 'read', clicked: 'clicked'})
//   }).then(res => {
//     console.log("Request complete! response:", res);
//   })
// }

// function getCookie(name) {
//   var cookieValue = null;
//   if (document.cookie && document.cookie != '') {
//       var cookies = document.cookie.split(';');
//       for (var i = 0; i < cookies.length; i++) {
//           var cookie = jQuery.trim(cookies[i]);
//           // Does this cookie string begin with the name we want?
//           if (cookie.substring(0, name.length + 1) == (name + '=')) {
//               cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
//               break;
//           }
//       }
//   }
//   return cookieValue;
// }