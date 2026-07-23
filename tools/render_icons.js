const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fs = require("fs");
const {
    FaNewspaper, FaDatabase, FaBroom, FaCogs, FaBrain, FaChartLine,
    FaSearch, FaExclamationTriangle, FaBalanceScale, FaLightbulb,
    FaProjectDiagram, FaRocket
} = require("react-icons/fa");
const { MdOutlineSpeed } = require("react-icons/md");

const icons = {
    newspaper: FaNewspaper,
    database: FaDatabase,
    broom: FaBroom,
    cogs: FaCogs,
    brain: FaBrain,
    chartline: FaChartLine,
    search: FaSearch,
    warning: FaExclamationTriangle,
    balance: FaBalanceScale,
    lightbulb: FaLightbulb,
    diagram: FaProjectDiagram,
    rocket: FaRocket,
    speed: MdOutlineSpeed,
};

async function renderIcon(name, Component, color) {
    const svgString = ReactDOMServer.renderToStaticMarkup(
        React.createElement(Component, { size: 256, color })
    );
    const buf = await sharp(Buffer.from(svgString)).resize(256, 256).png().toBuffer();
    fs.writeFileSync(`assets/${name}_${color}.png`, buf);
}

(async () => {
    const colors = ["FFFFFF", "028090", "013A40", "02C39A"];
    for (const [name, Component] of Object.entries(icons)) {
        for (const color of colors) {
            await renderIcon(name, Component, color);
        }
    }
    console.log("Icons rendered.");
})();