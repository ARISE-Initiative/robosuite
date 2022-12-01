$(document).ready(function () {
    $('a[href^="http://"], a[href^="https://"]').not('a[class*=internal]').attr('target', '_blank');
 });