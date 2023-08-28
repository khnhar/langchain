var initFlag = true;
var randomIndex = 0;
const chatMsg = document.getElementById('chatMessage');

/* 보내기 버튼 클릭 시 */
async function sendMessage() {
    console.log("sendMessage 호출");
    if (initFlag) {
        document.getElementById('chatIntro').style.display = 'none';
        initFlag = false;
    }
    const userInput = document.getElementById('userInput').value.trim();
    console.log("userInput:", userInput);

    if (userInput !== '') {
        // 사용자 입력 출력
        appendMessage('user', userInput, 'text'); // 메시지 타입 명시

        try {
            // 챗봇 답변 출력
            const answerData = await getAnswer(userInput);
            console.log("answerData : ", answerData);

            // 챗봇의 텍스트 답변을 출력합니다.
            appendMessage('bot', answerData, 'text');

            if (userInput === '경주에 대해 알려줘') {
                const imageContainer = document.getElementById('image-container');
                console.log("sendMessage^^imageContainer", imageContainer);
                console.log("if (imageContainer)", !!imageContainer);
                if (imageContainer) {
                    // 이미지 컨테이너를 출력하고 이미지를 가져와서 표시합니다.
                    appendMessage('bot', imageContainer, 'image');
                    getYoutubeAndDisplayImages(imageContainer);
                } else {
                    console.error("Element with ID 'image-container' not found.");
                }
            }

            if (userInput === '경주에 대해 알려줘') {
                // 이미지 메시지 추가
                const imageMessage = document.createElement('div');
                // 이미지 추가 로직을 여기에 작성
                // imageMessage.appendChild(image);
                appendMessage(imageMessage, '', 'image'); // 이미지를 채팅창에 추가
            }


        // 사용자 버튼 선택 출력
        const btnDiv = document.createElement('div');
        btnDiv.setAttribute('class', 'chat-button');
        btnDiv.setAttribute('id', 'chatBtn');

            // 도슨트영상, 역사퀴즈 버튼
            const btnArr = ["▶ 도슨트 영상", "▤ 한국 문화 퀴즈"];
            for (var i = 0; i < btnArr.length; i++) {
                const newBtn = setBtn(btnArr[i]);
                if (userInput === '경주에 대해 알려줘') {
                    if (i == 0) {
                        newBtn.addEventListener('click', async function () {
                            viewDst();
                            //appendSubButtons(btnDiv);
                        });
                    } else {
                        newBtn.addEventListener('click', async function () {
                            viewHidden();
                            // appendSubButtons(btnDiv);
                        });
                    }
                    btnDiv.appendChild(newBtn);
                }
            }

            setTimeout(() => appendMessage(btnDiv, ''), 500);
            //appendSubButtons(btnDiv);
        } catch (error) {
            console.error('Error:', error);
            appendMessage('bot', "죄송해요, 제가 잘 이해하지 못했어요. 다른 질문을 해주시겠어요?", 'text');
        }
    }
}


function appendSubButtons(parentDiv) {
    const SubButtonsDiv = document.createElement('div');
    SubButtonsDiv.setAttribute('class', 'chat-button');

    const btnArr2 = ["☞ 여행코스추천", "▣ 관광지도"];
    for (var i = 0; i < btnArr2.length; i++) {
        const subArr = setBtn(btnArr2[i]);
        if (i == 0) {
            subArr.addEventListener('click', function () {
                viewHidden2();
            });
        } else {
            subArr.addEventListener('click', function () {
                viewMap();
            });
        }
        SubButtonsDiv.appendChild(subArr);
    }
    parentDiv.appendChild(SubButtonsDiv);
}


function appendSubButtons2(parentDiv) {
    const SubButtonsDiv = document.createElement('div');
    SubButtonsDiv.setAttribute('class', 'chat-button');

    const btnArr2 = ["관광지 AI사진"];
    const subArr = setBtn(btnArr2);
    subArr.addEventListener('click', function () {
                stableDiffusion();
            });

    SubButtonsDiv.appendChild(subArr);
    parentDiv.appendChild(SubButtonsDiv);
}




/* 채팅 창에 메시지 출력 */
function appendMessage(obj, message, messageType) {
    const messageDiv = document.createElement('div');

    if (typeof (obj) == 'object') {
        chatMsg.appendChild(obj);
    } else {
        var typeClass = '';

        switch (obj) {
            case "user":
                typeClass = "chat-msg-user";
                break;
            case "bot":
                typeClass = "chat-msg-bot";
                break;
            case "userBtn":
                typeClass = "chat-msg-user-btn";
                break;
            default:
                break;
        }

        messageDiv.setAttribute('class', typeClass);

        if (messageType === 'image') {
            if (message === '관광지도') {
                // 채팅창에 맵을 출력하기 위한 div 엘리먼트 생성
                const mapDiv = document.createElement('div');
                mapDiv.style.width = '800px'; // 원하는 너비
                mapDiv.style.height = '500px'; // 원하는 높이
                chatMsg.appendChild(mapDiv);
                viewMap(mapDiv); // viewMap 함수로 맵 엘리먼트 추가
            } else { //Youtube
                const imageElement = document.createElement('img'); //유튜브 캡처
                imageElement.setAttribute('src', message); // 이미지 소스 설정
                imageElement.setAttribute('class', 'bot-image');
                // 이미지의 고정된 크기를 설정합니다.
                imageElement.setAttribute('width', '700'); // 원하는 너비
                imageElement.setAttribute('height', '400'); // 원하는 높이
                messageDiv.appendChild(imageElement); // 이미지 삽입
            }
        } else {
            messageDiv.textContent = message.trim();
            console.log("messageDiv.textContent_1:", messageDiv.textContent);
        }
        // 수정된 부분: 텍스트 길이에 따라 높이 자동 조절
        messageDiv.style.height = "auto"; // 초기 높이 설정
        chatMsg.appendChild(messageDiv);
        messageDiv.style.height = "fit-content"; // 텍스트 크기에 딱 맞게 조절
    }
    chatMsg.style.display = 'block';
    if (moveScroll(messageDiv.offsetHeight)) {
        messageDiv.scrollIntoView();
    }
}




// 채팅 메시지를 추가하는 함수
function addMessage(message, isUser) {
    const chatMsg = document.querySelector('.chat-msg-bot'); // .chat-msg-bot 요소를 선택합니다.
    const messageDiv = document.createElement('div'); // 새로운 <div> 요소를 생성합니다.

    // isUser에 따라 메시지 스타일을 설정합니다.
    if (isUser) {
        messageDiv.classList.add('chat-msg-user');
    } else {
        messageDiv.classList.add('chat-msg-bot');
    }

    // 메시지를 <div> 요소에 추가합니다.
    messageDiv.textContent = message.trim();
    console.log("messageDiv.textContent_2:",messageDiv.textContent)
    chatMsg.appendChild(messageDiv); // .chat-msg-bot 요소에 새로운 <div> 요소를 추가합니다.
}


/* 사용자가 선택할 버튼 생성 */
function setBtn(message, onClickFunction) {
    const btn = document.createElement('div');
    btn.setAttribute('class', "chat-button-sel");
    btn.textContent = message;
    //btn.addEventListener('click', onClickFunction); // 버튼 클릭 이벤트 추가
    return btn;
}



/* 채팅 창에 출력할 답변 목록 */
async function getAnswer(question) {
    const userInput = document.getElementById('userInput').value.trim();

    try {
        const response = await fetch('http://127.0.0.1:5000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput }),
            mode: 'cors'
        });

        const data = await response.json();
        const botMessage = data.bot_message; // 수정된 부분
        console.log('botMessage : ' + botMessage); // 수정된 부분
        return botMessage; // 수정된 부분
    } catch (error) {
        console.error('Error:', error);
        throw new Error("챗봇 응답을 가져오는 중 오류가 발생했습니다.");
    }
}



async function getYoutubeAndDisplayImages(imageContainer) {
    const data = {
        video_url: "https://youtu.be/dqkfpKJw348?si=UsnppEjLXD7mU7t3",
        search_query: "한복"
    };

    const requestOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/api/youTube', requestOptions);
        const responseData = await response.json();

        if (responseData.success) {
            const resultImagePaths = responseData.result_images_paths;

            if (imageContainer) {
                resultImagePaths.forEach(imagePath => {
                    const image = document.createElement('img');
                    image.src = imagePath;
                    image.style.maxWidth = '700';
                    image.style.maxHeight = '400';
                    image.style.margin = '10px';

                    // 이미지를 채팅창에 추가하는 로직
                    const imageMessage = document.createElement('div');
                    imageMessage.appendChild(image);
                    //appendMessage('bot', imageMessage, 'image'); // 이미지를 채팅창에 추가
                });
            } else {
                console.error("Element with ID 'image-container' not found.");
            }
        } else {
            console.error("Error:", responseData.error_message);
        }
    } catch (error) {
        console.error("Fetch error:", error);
    }
}






/* 도슨트 영상 */
function viewDst() {
    var videoDiv = document.createElement('div');
    videoDiv.setAttribute('class', "video-container");

    var iframeElement = document.createElement('iframe');
    iframeElement.src = "./dst.mp4";
    iframeElement.width = "400";
    iframeElement.height = "400";
    iframeElement.classList.add("video-iframe");

    videoDiv.appendChild(iframeElement);

    document.getElementById('chatMessage').appendChild(videoDiv);

    if (moveScroll(videoDiv.offsetHeight)) {
        videoDiv.scrollIntoView();
    }

    // 버튼 추가를 원하는 위치의 부모 엘리먼트를 선택하여 버튼 추가
    //var parentElementForButtons = document.getElementById('chatMessage'); // 적절한 엘리먼트 선택
    appendSubButtons(videoDiv);
}


/*버튼 눌렀을 때, GPT가 대답해 줌*/
async function viewHidden() {
    const userInput = "경주역사 문제와 보기는 4개 정답도 알려주는 스타일의 문제를 2개 만들어줘";

    try {
        //quiz
        const response = await fetch('http://127.0.0.1:5000/api/quiz', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput }),
            mode: 'cors'
        });
        const data = await response.json();
        const botMessage = data.bot_message;
        // 화면에 출력
        appendMessage('bot', botMessage, 'text');

    } catch (error) {
        console.error('Error:', error);
        throw new Error("챗봇 응답을 가져오는 중 오류가 발생했습니다.");
    }
}

async function viewHidden2() {
    const userInput = "경주 1박2일 여행코스를 알려주고, 인기있는 '여행 사이트명'과 url 3개 알려줘";

    try {
        //quiz
        const response = await fetch('http://127.0.0.1:5000/api/quiz', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput }),
            mode: 'cors'
        });
        const data = await response.json();
        const botMessage = data.bot_message;
        // 화면에 출력
        appendMessage('bot', botMessage, 'text');

    } catch (error) {
        console.error('Error:', error);
        throw new Error("챗봇 응답을 가져오는 중 오류가 발생했습니다.");
    }
}



async function stableDiffusion() {
    const apiUrl = 'http://127.0.0.1:5000/api/diffusion'; // 서버 URL을 적절하게 변경해주세요
    const requestData = {
        model_id: "stabilityai/stable-diffusion-2",
        prompt: "Please paint a watercolor of a photo of Gyeongju."
    };

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error('Error fetching image from the server');
        }

        const imageData = await response.text();
        appendMessage('bot', imageData, 'image');

    } catch (error) {
        console.error('Error:', error);
        // 에러 처리
    }
}




var quizIdx = 0;
function viewQuiz() {
    var quizArea = document.createElement('div');
    var quizDiv = document.getElementById('quiz-container');

    for(var i=0; i<3; i++) {
        var quiz = document.getElementById('quiz'+(quizIdx+1));
        if(i == quizIdx) {
            quiz.style.dispaly = "block";
        } else {
            quiz.style.dispaly = "none";
        }
    }
    quizArea.setAttribute('id', 'quiz-container');
    quizArea.innerHTML = quizDiv.innerHTML;

    const parentDiv = document.getElementById('chatMessage');
    // const preQuiz = parentDiv.getElementById('quiz-container');
    // preQuiz.remove();
    parentDiv.appendChild(quizArea);

    if(moveScroll(parentDiv.offsetHeight)) {
        parentDiv.scrollIntoView();
    }

    appendSubButtons(quizArea);
}



// '관광지도' 버튼 클릭 시
// show_map 함수를 호출하여 서버에서 생성된 HTML을 가져와서 채팅창에 표시하는 함수
async function viewMap() {
    const apiUrl = 'http://127.0.0.1:5000/api/googlemap';

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'}
        });
        console.log("viewMap_response::", response);

        if (response.ok) {
            const mapHtml = await response.text(); // mapHtml 정의
            const parser = new DOMParser();
            const mapHtmlElement = parser.parseFromString(mapHtml, 'text/html').body.firstChild;

            // 채팅창에 맵을 출력하기 위한 div 엘리먼트 생성
            const mapDiv = document.createElement('div');
            mapDiv.setAttribute('class', "video-container");
            mapDiv.appendChild(mapHtmlElement); // 이미 HTML 엘리먼트로 변환되었으므로 바로 추가합니다.
            mapDiv.style.width = '825px'; // 원하는 너비
            mapDiv.style.height = '500px'; // 원하는 높이

            // 맵을 채팅창에 추가
            const chatMessage = document.getElementById('chatMessage');
            chatMessage.appendChild(mapDiv);

            if (moveScroll(mapDiv.offsetHeight)) {
                mapDiv.scrollIntoView();
            }
            appendSubButtons2(mapDiv);

        } else {
            console.error('Error:', response.statusText);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}



function goAnswer(answerYn) {
    var answers = ['N', 'Y', 'Y'];
    if(answers[0] == answerYn) {
        alert("와우! 정답이에요~");
    } else {
        alert("안타깝지만 틀렸어요.");
    }
    // quizIdx = quizIdx + 1;

    // viewQuiz();
}



/* 채팅창 스크롤 이동 */
function moveScroll(addHeight) {
    chatMsg.scrollTop = chatMsg.scrollHeight + addHeight; // 스크롤 맨 아래로 이동
    return true;
}
