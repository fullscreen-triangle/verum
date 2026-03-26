import styled from 'styled-components'

function InfoPanel(){
    const [isVisible, setIsVisible] = React.useState(true)
    const exitRef = React.useRef(null)


function clickHandler(e){
    setIsVisible(false)
}

    return(
        <Container style={{display: isVisible ? "flex" : "none"}}>
            <Button onClick={(() => setIsVisible(false))}>X</Button>
            <Header>
                Montreal's Most Dangerous Roads
            </Header>
            <SubHeader>
                All Reported Road Accidents in Montreal, Quebec 
                from 2012 to 2018
            </SubHeader>
            <Content>
                Accidents that resulted in injuries have a glow. 
                A larger number or higher severity accidents have a larger 
                glow. <br /><br />
                Dots with a yellow appearace have minor or no injuries, orange have major injuries, and red have deaths.
            </Content>
            <Source>
            Data obtained from <a href="https://www.donneesquebec.ca/recherche/fr/dataset/vmtl-collisions-routieres"><span style={{fontStyle:"italic", color: "#CBE9F6"}}>Quebec Open Data Hub</span>.</a>
            </Source>
            <h5 style={{color: "#3A7095"}}>Total Number of Accidents:</h5>
            <SummaryBox>171.2k</SummaryBox>
        </Container>
    )
}

export default InfoPanel

const Container = styled.div`
z-index:10;
display:flex;
flex-direction: column;
background: #09121F;
position: absolute;
top: 0;
margin:10px 10px 0 0 ;
right: 0;
padding: 30px;
width: 25vw;
border-radius: 5px;
color: #CBE9F6;
`
const Header = styled.h1`
font-weight: bold;
font-size: 1.4em;
margin: 0;
`
const SubHeader = styled.h2`
font-size: 0.8em;
`
const Content = styled.h4`
font-size: 0.6em;
color: rgb(75,145,193);
`
const Source = styled.h6`
font-size: 0.6em;
margin: 0;
color:rgb(75,145,193);
`
const SummaryBox = styled.h1`
font-weight: 700;
margin: 0;
line-height: 0;
color: rgb(151,182,143)
`
const Button = styled.button`
color: red;
font-size: 1.3em;
font-weight: bold;
background: transparent;    
position: absolute;
right: 10px;
top: 10px;
border: 0px;
display: flex;
justify-content: flex-end;
margin: 0;

&:hover{
    cursor: pointer;
}
`