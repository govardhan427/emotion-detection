let emotionHistory = [];
const maxHistory = 5;

function updateEmotionList(emotion) {
    if (emotionHistory.length >= maxHistory) {
        emotionHistory.shift();
    }
    emotionHistory.push(emotion);

    const emotionList = document.getElementById("emotion-list");
    if (emotionList) {
        emotionList.innerHTML = "";
        emotionHistory.forEach((emo) => {
            let li = document.createElement("li");
            li.textContent = emo;
            emotionList.appendChild(li);
        });
    }

    localStorage.setItem("emotionData", JSON.stringify(emotionHistory));
}

async function captureAndPredict() {
    try {
        const response = await fetch("/predict", { method: "POST" });
        const data = await response.json();
        document.getElementById("result").textContent = `Detected: ${data.emotion}`;
        updateEmotionList(data.emotion);
    } catch (error) {
        console.error("Error detecting emotion:", error);
    }
}

// 📸 Image Preview for Uploads
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("image-input");
    const preview = document.getElementById("image-preview");

    if (input && preview) {
        input.addEventListener("change", function () {
            const file = this.files[0];

            if (file) {
                const reader = new FileReader();

                reader.addEventListener("load", function () {
                    preview.setAttribute("src", this.result);
                    preview.style.display = "block";
                });

                reader.readAsDataURL(file);
            } else {
                preview.style.display = "none";
                preview.removeAttribute("src");
            }
        });
    }
});