// Clock

setInterval(() => {

    document.getElementById("clock").innerHTML =
        new Date().toLocaleString();

}, 1000);


// Snapshot

document
.getElementById("snapshot-btn")
.addEventListener("click", () => {

    fetch("/snapshot/")
    .then(res => res.json())
    .then(data => {

        showToast(data.message || 'Snapshot captured successfully!');

    });

});


// Face Count

setInterval(() => {

    fetch("/face_count/")
    .then(res => res.json())
    .then(data => {

        document.getElementById("face-count").innerHTML =
            data.count;

    });

}, 1000);


// People Count

setInterval(() => {

    fetch("/people_count/")
    .then(res => res.json())
    .then(data => {

        document.getElementById("people-count").innerHTML =
            data.count;

    });

}, 1000);